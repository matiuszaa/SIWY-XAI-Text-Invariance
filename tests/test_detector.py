from pathlib import Path

import numpy as np
import pytest

from xai_inv.config import load_config
from xai_inv.detector import Detector

SMOKE = load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")
TEXTS = [
    "The gold standard is a very outdated concept and is no longer used .",
    "There are several reasons why you might feel thirsty.\n\nFirst, dehydration.",
    "Short.",
]


@pytest.fixture(scope="module")
def detector():
    return Detector(SMOKE.detector.model, device=SMOKE.detector.device, batch_size=2)


def test_predict_shapes_and_range(detector):
    pred = detector.predict(TEXTS)
    assert pred.labels.shape == pred.probs.shape == (len(TEXTS),)
    assert set(pred.labels) <= {0, 1}
    assert ((pred.probs >= 0) & (pred.probs <= 1)).all()
    np.testing.assert_array_equal(pred.labels, (pred.probs > 0.5).astype(int))


def test_predict_independent_of_batching(detector):
    batched = detector.predict(TEXTS).probs
    single = np.concatenate([detector.predict([t]).probs for t in TEXTS])
    np.testing.assert_allclose(batched, single, atol=1e-5)


def test_predict_empty(detector):
    assert len(detector.predict([]).labels) == 0
