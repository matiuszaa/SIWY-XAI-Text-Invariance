from simalign import SentenceAligner

METHODS = {"inter": "a", "mwmf": "m", "itermax": "i"}
MAX_SUBWORDS = 510


def targets_per_word(edges: list[tuple[int, int]], n_words: int) -> list[list[int]]:
    targets = [[] for _ in range(n_words)]
    for i, j in sorted(edges):
        targets[i].append(j)
    return targets


def alignment_coverage(alignment: list[list[int]]) -> float:
    if len(alignment) == 0:
        return float("nan")
    return sum(len(targets) > 0 for targets in alignment) / len(alignment)


class WordAligner:
    def __init__(self, model: str, method: str = "itermax", device: str = "cpu"):
        if method not in METHODS:
            raise ValueError(f"Unknown alignment method: {method}")
        self.method = method
        self.aligner = SentenceAligner(
            model=model, token_type="bpe", matching_methods=METHODS[method], device=device
        )

    def _check_length(self, words: list[str]) -> None:
        tokenizer = self.aligner.embed_loader.tokenizer
        n_subwords = sum(len(tokenizer.tokenize(word)) for word in words)
        if n_subwords > MAX_SUBWORDS:
            raise ValueError(f"Text has {n_subwords} subwords, simalign supports {MAX_SUBWORDS}")

    def align(self, original: str, transformed: str) -> list[list[int]]:
        source, target = original.split(), transformed.split()
        self._check_length(source)
        self._check_length(target)
        edges = self.aligner.get_word_aligns(source, target)[self.method]
        return targets_per_word(edges, len(source))
