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


def test_transform_synonyms_writes_pairs(tmp_path, monkeypatch):
    pd.DataFrame(
        {"text_id": ["a", "b"], "label": [0, 1], "text": ["A small cat.", "A calm dog."]}
    ).to_parquet(tmp_path / "texts.parquet", index=False)
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "paths": {"interim": str(tmp_path)},
            "transforms": {"synonyms": {"fraction": 0.1}},
        }
    )
    monkeypatch.setattr(
        transforms,
        "build_synonym_augmenter",
        lambda fraction: FakeAugmenter(["A little cat."]),
    )

    out_path = transforms.transform_synonyms(cfg)
    pairs = pd.read_parquet(out_path)

    assert out_path == tmp_path / "pairs.parquet"
    assert pairs.pair_id.tolist() == ["a-synonyms", "b-synonyms"]
    assert pairs.transformation_status.tolist() == ["ok", "ok"]
    assert [alignment.tolist() for alignment in pairs.word_alignment] == [[0, 1, 2], [0, 1, 2]]
