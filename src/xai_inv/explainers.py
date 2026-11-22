from typing import NamedTuple

import numpy as np
from ferret import IntegratedGradientExplainer, LIMEExplainer, SHAPExplainer

from xai_inv.detector import Detector


class WordAttribution(NamedTuple):
    words: list[str]
    scores: np.ndarray
    nonword_score: float


def token_word_ids(text: str, offsets: list[tuple[int, int]]) -> list[int | None]:
    char_word = np.full(len(text), -1)
    pos = 0
    for i, word in enumerate(text.split()):
        pos = text.index(word, pos)
        char_word[pos : pos + len(word)] = i
        pos += len(word)
    word_ids = []
    for start, end in offsets:
        if start == end or text[start:end].isspace():
            word_ids.append(None)
            continue
        word_id = int(char_word[start])
        if word_id < 0:
            raise ValueError(f"Token at {start}:{end} does not start inside a word")
        word_ids.append(word_id)
    return word_ids


def aggregate_to_words(text: str, offsets, token_scores) -> WordAttribution:
    words = text.split()
    word_ids = token_word_ids(text, offsets)
    if len(word_ids) != len(token_scores):
        raise ValueError(f"{len(word_ids)} tokens but {len(token_scores)} scores")
    scores = np.zeros(len(words))
    nonword = 0.0
    for word_id, score in zip(word_ids, token_scores, strict=True):
        if word_id is None:
            nonword += float(score)
        else:
            scores[word_id] += score
    if len(set(word_ids) - {None}) < len(words):
        raise ValueError("Some words got no tokens")
    return WordAttribution(words, scores, nonword)


EXPLAINERS = {
    "ig": IntegratedGradientExplainer,
    "lime": LIMEExplainer,
    "shap": SHAPExplainer,
}


def explain(
    detector: Detector, method: str, text: str, target: int, **explainer_args
) -> WordAttribution:
    if method not in EXPLAINERS:
        raise ValueError(f"Unknown explainer: {method}")
    explainer = EXPLAINERS[method](detector.model, detector.tokenizer)
    explanation = explainer(text, target=target, **explainer_args)
    offsets = detector.tokenizer(text, truncation=True, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    return aggregate_to_words(text, offsets, explanation.scores)


def explain_ig(detector: Detector, text: str, target: int, n_steps: int = 50) -> WordAttribution:
    return explain(detector, "ig", text, target, call_args={"n_steps": n_steps})


def explain_lime(
    detector: Detector, text: str, target: int, n_samples: int = 1000, seed: int = 42
) -> WordAttribution:
    return explain(
        detector,
        "lime",
        text,
        target,
        init_args={"random_state": seed},
        call_args={"num_samples": n_samples, "batch_size": detector.batch_size},
    )


def explain_shap(
    detector: Detector, text: str, target: int, max_evals: int = 500, seed: int = 42
) -> WordAttribution:
    return explain(
        detector,
        "shap",
        text,
        target,
        init_args={"seed": seed},
        call_args={"max_evals": max_evals, "batch_size": detector.batch_size},
    )
