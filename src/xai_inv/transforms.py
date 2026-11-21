import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

WORD_TOKEN = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
MAX_NEW_TOKENS = 256


class Augmenter(Protocol):
    def augment(self, text: str) -> list[str]: ...


class WordNetSynonymAugmenter:
    def __init__(self, fraction: float):
        self.fraction = fraction

    @staticmethod
    def _candidates(word: str) -> list[str]:
        from nltk.corpus import wordnet

        try:
            synsets = wordnet.synsets(word)
        except LookupError:
            import nltk

            if not nltk.download("wordnet", quiet=True):
                raise RuntimeError("Could not download the NLTK WordNet corpus") from None
            synsets = wordnet.synsets(word)

        candidates = {
            lemma.name()
            for synset in synsets
            for lemma in synset.lemmas()
            if lemma.name().isalpha() and lemma.name().lower() != word.lower()
        }
        return sorted(candidates)

    def augment(self, text: str) -> list[str]:
        tokens = text.split()
        eligible = []
        for index, token in enumerate(tokens):
            match = WORD_TOKEN.match(token)
            if match is None or len(match.group(2)) < 3:
                continue
            candidates = self._candidates(match.group(2))
            if candidates:
                eligible.append((index, match, candidates))

        if not eligible:
            return []

        n_replacements = min(len(eligible), max(1, round(len(eligible) * self.fraction)))
        for index, match, candidates in random.sample(eligible, n_replacements):
            replacement = random.choice(candidates)
            original_word = match.group(2)
            if original_word.isupper():
                replacement = replacement.upper()
            elif original_word.istitle():
                replacement = replacement.title()
            tokens[index] = f"{match.group(1)}{replacement}{match.group(3)}"
        return [" ".join(tokens)]


def split_sentences(text: str) -> list[list[str]]:
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if not nltk.download("punkt_tab", quiet=True):
            raise RuntimeError("Could not download the NLTK punkt_tab tokenizer") from None
    paragraphs = [" ".join(p.split()) for p in PARAGRAPH_BREAK.split(text)]
    return [nltk.sent_tokenize(p) for p in paragraphs if p]


def rewrite_sentences(text: str, rewrite: Callable[[list[str]], list[str]]) -> str:
    paragraphs = split_sentences(text)
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
    rewritten = rewrite(sentences)
    if len(rewritten) != len(sentences):
        raise ValueError(f"Got {len(rewritten)} rewritten sentences for {len(sentences)}")
    it = iter(rewritten)
    return "\n\n".join(" ".join(next(it) for _ in paragraph) for paragraph in paragraphs)


class Seq2SeqGenerator:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        prefix: str = "",
        **generate,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.prefix = prefix
        self.generate = generate
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, texts: list[str]) -> list[str]:
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                [self.prefix + text for text in texts[start : start + self.batch_size]],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            generated = self.model.generate(**batch, **self.generate)
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(text.strip() for text in decoded)
        return outputs


class BackTranslationAugmenter:
    def __init__(
        self,
        forward: Callable[[list[str]], list[str]],
        backward: Callable[[list[str]], list[str]],
    ):
        self.forward = forward
        self.backward = backward

    def augment(self, text: str) -> list[str]:
        return [rewrite_sentences(text, lambda sentences: self.backward(self.forward(sentences)))]


class ParaphraseAugmenter:
    def __init__(self, paraphrase: Callable[[list[str]], list[str]]):
        self.paraphrase = paraphrase

    def augment(self, text: str) -> list[str]:
        return [rewrite_sentences(text, self.paraphrase)]


PAIR_COLUMNS = [
    "pair_id",
    "text_id",
    "label",
    "transformation",
    "original_text",
    "transformed_text",
    "original_n_words",
    "transformed_n_words",
    "word_alignment",
    "transformation_status",
    "transformation_reason",
]


def build_synonym_augmenter(fraction: float) -> Augmenter:
    if not 0 < fraction <= 1:
        raise ValueError(f"synonyms.fraction must be in (0, 1], got {fraction}")
    return WordNetSynonymAugmenter(fraction)


def synonym_pair(text: str, augmenter: Augmenter) -> dict[str, object]:
    original_words = text.split()
    candidates = augmenter.augment(text)
    if not candidates:
        return {
            "transformed_text": None,
            "transformed_n_words": None,
            "word_alignment": None,
            "transformation_status": "rejected",
            "transformation_reason": "no_candidate",
        }

    transformed = candidates[0].strip()
    transformed_words = transformed.split()
    if transformed == text:
        reason = "unchanged"
    elif len(transformed_words) != len(original_words):
        reason = "word_count_changed"
    else:
        return {
            "transformed_text": transformed,
            "transformed_n_words": len(transformed_words),
            "word_alignment": list(range(len(original_words))),
            "transformation_status": "ok",
            "transformation_reason": None,
        }

    return {
        "transformed_text": transformed,
        "transformed_n_words": len(transformed_words),
        "word_alignment": None,
        "transformation_status": "rejected",
        "transformation_reason": reason,
    }


def rewrite_pair(text: str, augmenter: Augmenter) -> dict[str, object]:
    candidates = augmenter.augment(text)
    transformed = candidates[0].strip() if candidates else ""
    if not transformed:
        status, reason = "rejected", "no_candidate"
    elif transformed.split() == text.split():
        status, reason = "rejected", "unchanged"
    else:
        status, reason = "ok", None
    return {
        "transformed_text": transformed or None,
        "transformed_n_words": len(transformed.split()) if transformed else None,
        "word_alignment": None,
        "transformation_status": status,
        "transformation_reason": reason,
    }


PAIR_BUILDERS = {
    "synonyms": synonym_pair,
    "backtranslation": rewrite_pair,
    "paraphrase": rewrite_pair,
}


def build_generator(model_name: str, params: DictConfig, prefix: str = "") -> Seq2SeqGenerator:
    generate = {"max_new_tokens": MAX_NEW_TOKENS, **OmegaConf.to_container(params.generate)}
    return Seq2SeqGenerator(
        model_name,
        device=params.device,
        batch_size=int(params.batch_size),
        prefix=prefix,
        **generate,
    )


def build_augmenter(name: str, params: DictConfig) -> Augmenter:
    if name == "synonyms":
        return build_synonym_augmenter(float(params.fraction))
    if name == "backtranslation":
        return BackTranslationAugmenter(
            build_generator(params.forward_model, params),
            build_generator(params.backward_model, params),
        )
    if name == "paraphrase":
        return ParaphraseAugmenter(build_generator(params.model, params, prefix=params.prefix))
    raise ValueError(f"Unknown transformation: {name}")


def transform_texts(cfg: DictConfig) -> Path:
    interim_dir = Path(cfg.paths.interim)
    texts_path = interim_dir / "texts.parquet"
    if not texts_path.is_file():
        raise FileNotFoundError(f"Prepared texts not found: {texts_path}. Run 'prepare' first.")

    texts = pd.read_parquet(texts_path)
    required_columns = {"text_id", "label", "text"}
    missing_columns = required_columns - set(texts.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Prepared texts missing required columns: {missing}")

    rows = []
    for name, params in cfg.transforms.items():
        random.seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        augmenter = build_augmenter(name, params)
        make_pair = PAIR_BUILDERS[name]
        for row in texts.itertuples(index=False):
            rows.append(
                {
                    "pair_id": f"{row.text_id}-{name}",
                    "text_id": row.text_id,
                    "label": int(row.label),
                    "transformation": name,
                    "original_text": row.text,
                    "original_n_words": len(row.text.split()),
                    **make_pair(row.text, augmenter),
                }
            )

    out_path = interim_dir / "pairs.parquet"
    pd.DataFrame(rows, columns=PAIR_COLUMNS).to_parquet(out_path, index=False)
    return out_path
