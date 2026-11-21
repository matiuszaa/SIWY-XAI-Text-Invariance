import random

import pandas as pd
import pytest
from omegaconf import OmegaConf

from xai_inv import transforms


class FakeAugmenter:
    def __init__(self, result):
        self.result = result

    def augment(self, text):
        return self.result


def test_synonym_pair_keeps_identity_alignment_for_same_length_result():
    pair = transforms.synonym_pair("A small cat sleeps.", FakeAugmenter(["A little cat sleeps."]))
    assert pair == {
        "transformed_text": "A little cat sleeps.",
        "transformed_n_words": 4,
        "word_alignment": [0, 1, 2, 3],
        "transformation_status": "ok",
        "transformation_reason": None,
    }


def test_wordnet_augmenter_preserves_punctuation_and_word_count(monkeypatch):
    augmenter = transforms.WordNetSynonymAugmenter(fraction=1.0)
    monkeypatch.setattr(
        augmenter,
        "_candidates",
        lambda word: ["little"] if word.lower() == "small" else [],
    )

    assert augmenter.augment("A small cat.") == ["A little cat."]


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        ([], "no_candidate"),
        (["A small cat sleeps."], "unchanged"),
        (["A very tiny feline sleeps."], "word_count_changed"),
    ],
)
def test_synonym_pair_rejects_unsafe_results(result, reason):
    pair = transforms.synonym_pair("A small cat sleeps.", FakeAugmenter(result))
    assert pair["transformation_status"] == "rejected"
    assert pair["transformation_reason"] == reason
    assert pair["word_alignment"] is None


def test_rewrite_sentences_keeps_paragraphs():
    text = "First one. Second one.\n\nThird\n one."
    assert transforms.split_sentences(text) == [["First one.", "Second one."], ["Third one."]]
    result = transforms.rewrite_sentences(text, lambda sentences: [s.upper() for s in sentences])
    assert result == "FIRST ONE. SECOND ONE.\n\nTHIRD ONE."


def test_rewrite_sentences_rejects_missing_output():
    with pytest.raises(ValueError, match="rewritten sentences"):
        transforms.rewrite_sentences("One. Two.", lambda sentences: sentences[:1])


def test_back_translation_goes_through_pivot_and_back():
    calls = []

    def forward(sentences):
        calls.append(("forward", sentences))
        return [f"de:{s}" for s in sentences]

    def backward(sentences):
        calls.append(("backward", sentences))
        return [s.removeprefix("de:").replace("cat", "kitten") for s in sentences]

    augmenter = transforms.BackTranslationAugmenter(forward, backward)

    assert augmenter.augment("A cat sleeps. It is warm.") == ["A kitten sleeps. It is warm."]
    assert calls == [
        ("forward", ["A cat sleeps.", "It is warm."]),
        ("backward", ["de:A cat sleeps.", "de:It is warm."]),
    ]


def test_rewrite_pair_has_no_alignment_yet():
    pair = transforms.rewrite_pair("A cat sleeps.", FakeAugmenter(["The cat is asleep."]))
    assert pair == {
        "transformed_text": "The cat is asleep.",
        "transformed_n_words": 4,
        "word_alignment": None,
        "transformation_status": "ok",
        "transformation_reason": None,
    }


@pytest.mark.parametrize(
    ("result", "reason"),
    [([], "no_candidate"), ([" "], "no_candidate"), (["A  cat sleeps. "], "unchanged")],
)
def test_rewrite_pair_rejects_empty_and_unchanged(result, reason):
    pair = transforms.rewrite_pair("A cat sleeps.", FakeAugmenter(result))
    assert pair["transformation_status"] == "rejected"
    assert pair["transformation_reason"] == reason


def test_paraphrase_rewrites_every_sentence():
    augmenter = transforms.ParaphraseAugmenter(lambda sentences: [f"<{s}>" for s in sentences])
    assert augmenter.augment("One. Two.\n\nThree.") == ["<One.> <Two.>\n\n<Three.>"]


@pytest.fixture
def created_generators(monkeypatch):
    created = []

    class FakeGenerator:
        def __init__(self, model_name, **kwargs):
            created.append((model_name, kwargs))

    monkeypatch.setattr(transforms, "Seq2SeqGenerator", FakeGenerator)
    return created


def test_build_augmenter_back_translation(created_generators):
    params = OmegaConf.create(
        {
            "forward_model": "en-de",
            "backward_model": "de-en",
            "batch_size": 4,
            "device": "cpu",
            "generate": {"num_beams": 2},
        }
    )

    augmenter = transforms.build_augmenter("backtranslation", params)

    assert isinstance(augmenter, transforms.BackTranslationAugmenter)
    assert [name for name, _ in created_generators] == ["en-de", "de-en"]
    assert created_generators[0][1] == {
        "device": "cpu",
        "batch_size": 4,
        "prefix": "",
        "num_beams": 2,
        "max_new_tokens": transforms.MAX_NEW_TOKENS,
    }


def test_build_augmenter_paraphrase(created_generators):
    params = OmegaConf.create(
        {
            "model": "t5",
            "prefix": "paraphrase: ",
            "batch_size": 8,
            "device": "cpu",
            "generate": {"num_beams": 5, "no_repeat_ngram_size": 2, "max_new_tokens": 64},
        }
    )

    augmenter = transforms.build_augmenter("paraphrase", params)

    assert isinstance(augmenter, transforms.ParaphraseAugmenter)
    assert created_generators == [
        (
            "t5",
            {
                "device": "cpu",
                "batch_size": 8,
                "prefix": "paraphrase: ",
                "num_beams": 5,
                "no_repeat_ngram_size": 2,
                "max_new_tokens": 64,
            },
        )
    ]


@pytest.fixture
def texts_dir(tmp_path):
    pd.DataFrame(
        {"text_id": ["a", "b"], "label": [0, 1], "text": ["A small cat.", "A calm dog."]}
    ).to_parquet(tmp_path / "texts.parquet", index=False)
    return tmp_path


def make_cfg(path, transforms_cfg):
    return OmegaConf.create(
        {"seed": 42, "paths": {"interim": str(path)}, "transforms": transforms_cfg}
    )


def test_transform_texts_writes_pairs(texts_dir, monkeypatch):
    cfg = make_cfg(texts_dir, {"synonyms": {"fraction": 0.1}})
    monkeypatch.setattr(
        transforms,
        "build_synonym_augmenter",
        lambda fraction: FakeAugmenter(["A little cat."]),
    )

    out_path = transforms.transform_texts(cfg)
    pairs = pd.read_parquet(out_path)

    assert out_path == texts_dir / "pairs.parquet"
    assert pairs.pair_id.tolist() == ["a-synonyms", "b-synonyms"]
    assert pairs.transformation_status.tolist() == ["ok", "ok"]
    assert [alignment.tolist() for alignment in pairs.word_alignment] == [[0, 1, 2], [0, 1, 2]]


def test_transform_texts_runs_every_configured_transformation(texts_dir, monkeypatch):
    cfg = make_cfg(texts_dir, {"synonyms": {"fraction": 0.1}, "other": {}})
    monkeypatch.setattr(transforms, "build_augmenter", lambda name, params: FakeAugmenter([]))
    monkeypatch.setitem(transforms.PAIR_BUILDERS, "other", transforms.synonym_pair)

    pairs = pd.read_parquet(transforms.transform_texts(cfg))

    assert pairs.pair_id.tolist() == ["a-synonyms", "b-synonyms", "a-other", "b-other"]
    assert pairs.transformation.tolist() == ["synonyms", "synonyms", "other", "other"]


class RandomAugmenter:
    def augment(self, text):
        return [f"{text} {random.random():.6f}"]


def count_builds(monkeypatch):
    builds = []

    def build(name, params):
        builds.append(name)
        return RandomAugmenter()

    monkeypatch.setattr(transforms, "build_augmenter", build)
    monkeypatch.setitem(transforms.PAIR_BUILDERS, "noise", transforms.rewrite_pair)
    return builds


def test_transform_texts_reuses_cached_pairs(texts_dir, monkeypatch):
    builds = count_builds(monkeypatch)
    cfg = make_cfg(texts_dir, {"noise": {"level": 1}})

    first = pd.read_parquet(transforms.transform_texts(cfg))
    second = pd.read_parquet(transforms.transform_texts(cfg))

    assert builds == ["noise"]
    pd.testing.assert_frame_equal(first, second)


def test_transform_texts_recomputes_after_config_change(texts_dir, monkeypatch):
    builds = count_builds(monkeypatch)
    transforms.transform_texts(make_cfg(texts_dir, {"noise": {"level": 1}}))
    pairs = pd.read_parquet(
        transforms.transform_texts(make_cfg(texts_dir, {"noise": {"level": 2}}))
    )

    assert builds == ["noise", "noise"]
    assert pairs.transform_hash.nunique() == 1


def test_transform_texts_resumed_run_matches_fresh_run(texts_dir, monkeypatch):
    count_builds(monkeypatch)
    cfg = make_cfg(texts_dir, {"noise": {"level": 1}})
    fresh = pd.read_parquet(transforms.transform_texts(cfg))
    fresh.iloc[:1].to_parquet(texts_dir / "pairs.parquet", index=False)

    resumed = pd.read_parquet(transforms.transform_texts(cfg))

    pd.testing.assert_frame_equal(fresh, resumed)


def test_transform_texts_rejects_unknown_transformation(texts_dir):
    with pytest.raises(ValueError, match="Unknown transformation"):
        transforms.transform_texts(make_cfg(texts_dir, {"typos": {}}))
