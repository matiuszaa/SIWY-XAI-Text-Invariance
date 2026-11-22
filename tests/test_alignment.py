import math

import pytest

from xai_inv import alignment


def test_targets_per_word_keeps_unaligned_and_multiple_targets():
    edges = [(2, 1), (0, 0), (2, 3)]
    assert alignment.targets_per_word(edges, 4) == [[0], [], [1, 3], []]


def test_alignment_coverage():
    assert alignment.alignment_coverage([[0], [], [1, 3], []]) == 0.5
    assert math.isnan(alignment.alignment_coverage([]))


class FakeTokenizer:
    def tokenize(self, word):
        return list(word)


class FakeSentenceAligner:
    def __init__(self, model, token_type, matching_methods, device):
        assert (model, token_type, matching_methods, device) == ("mbert", "bpe", "i", "cpu")
        self.embed_loader = type("Loader", (), {"tokenizer": FakeTokenizer()})()

    def get_word_aligns(self, source, target):
        assert source == ["a", "big", "cat"]
        assert target == ["the", "cat", "is", "large"]
        return {"itermax": [(2, 1), (1, 3)]}


@pytest.fixture
def aligner(monkeypatch):
    monkeypatch.setattr(alignment, "SentenceAligner", FakeSentenceAligner)
    return alignment.WordAligner("mbert", method="itermax", device="cpu")


def test_word_aligner_maps_original_words_to_transformed_words(aligner):
    assert aligner.align("a big cat", "the cat is large") == [[], [3], [1]]


def test_word_aligner_rejects_texts_longer_than_model_input(aligner, monkeypatch):
    monkeypatch.setattr(alignment, "MAX_SUBWORDS", 5)
    with pytest.raises(ValueError, match="subwords"):
        aligner.align("a big cat", "the cat is large")


def test_word_aligner_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown alignment method"):
        alignment.WordAligner("mbert", method="fwd")
