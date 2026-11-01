from typing import NamedTuple

import numpy as np
from ferret import IntegratedGradientExplainer

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


def explain_ig(detector: Detector, text: str, target: int, n_steps: int = 50) -> WordAttribution:
    explainer = IntegratedGradientExplainer(detector.model, detector.tokenizer)
    explanation = explainer(text, target=target, call_args={"n_steps": n_steps})
    offsets = detector.tokenizer(text, truncation=True, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    return aggregate_to_words(text, offsets, explanation.scores)
