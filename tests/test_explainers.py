from pathlib import Path

import numpy as np
import pytest

from xai_inv.config import load_config
from xai_inv.detector import Detector
from xai_inv.explainers import aggregate_to_words, explain_ig, token_word_ids

SMOKE = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")


def test_token_word_ids_handles_repeats_and_newlines():
    text = "it is\n\nit ,"
    offsets = [(0, 0), (0, 2), (3, 5), (5, 6), (6, 7), (7, 9), (10, 11), (0, 0)]
    assert token_word_ids(text, offsets) == [None, 0, 1, None, None, 2, 3, None]


def test_aggregate_sums_subtokens():
    text = "Decriminalized now."
    offsets = [(0, 0), (0, 3), (3, 10), (10, 14), (15, 18), (18, 19), (0, 0)]
    scores = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 0.25]
    att = aggregate_to_words(text, offsets, scores)
    assert att.words == ["Decriminalized", "now."]
    np.testing.assert_allclose(att.scores, [6.0, 9.0])
    assert att.nonword_score == 0.75


def test_aggregate_rejects_length_mismatch():
    with pytest.raises(ValueError):
        aggregate_to_words("a b", [(0, 0), (0, 1), (2, 3), (0, 0)], [1.0, 2.0])


@pytest.fixture(scope="module")
def detector():
    return Detector(SMOKE.detector.model, device=SMOKE.detector.device)


def test_explain_ig_word_level(detector):
    text = "When something is decriminalized , it wo n't go on your record.\n\nYou pay a fine."
    att = explain_ig(detector, text, target=1, n_steps=SMOKE.explainers.ig.n_steps)
    assert att.words == text.split()
    assert att.scores.shape == (len(att.words),)
    assert np.isfinite(att.scores).all()
    assert np.isfinite(att.nonword_score)
    no_newline = explain_ig(detector, text.replace("\n\n", " "), target=1, n_steps=5)
    assert no_newline.nonword_score == 0.0
