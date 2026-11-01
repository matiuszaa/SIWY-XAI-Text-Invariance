import json

import pytest

from xai_inv.data import load_answers, sample_texts, sha256_texts


@pytest.fixture
def hc3_file(tmp_path):
    items = [
        {
            "question": f"q{i}",
            "human_answers": [" ".join(["human"] * (40 + i))],
            "chatgpt_answers": [" ".join(["bot"] * (40 + i)) + f" {i}"],
            "index": "None",
            "source": "reddit_eli5",
        }
        for i in range(30)
    ]
    path = tmp_path / "all.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    return path


def test_load_answers(hc3_file):
    answers = load_answers(hc3_file)
    assert len(answers) == 60
    assert set(answers.label) == {0, 1}
    assert answers.text_id.is_unique


def test_sample_texts_balanced_and_in_range(hc3_file):
    texts = sample_texts(load_answers(hc3_file), n_texts=5, min_words=50, max_words=60, seed=42)
    assert len(texts) == 5
    assert texts.label.value_counts().to_dict() == {0: 2, 1: 3}
    assert texts.n_words.between(50, 60).all()


def test_sample_texts_deterministic(hc3_file):
    answers = load_answers(hc3_file)
    a = sample_texts(answers, n_texts=6, min_words=50, max_words=60, seed=1)
    b = sample_texts(answers, n_texts=6, min_words=50, max_words=60, seed=1)
    assert sha256_texts(a) == sha256_texts(b)


def test_sample_texts_not_enough(hc3_file):
    with pytest.raises(ValueError):
        sample_texts(load_answers(hc3_file), n_texts=100, min_words=50, max_words=60, seed=0)
