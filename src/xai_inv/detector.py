from typing import NamedTuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHATGPT = 1


class Predictions(NamedTuple):
    labels: np.ndarray
    probs: np.ndarray


class Detector:
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        if self.model.config.num_labels != 2:
            raise ValueError(f"Expected a binary detector, got {self.model.config.num_labels}")

    @torch.inference_mode()
    def predict(self, texts: list[str]) -> Predictions:
        probs = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**batch).logits
            probs.append(logits.softmax(-1)[:, CHATGPT].float().cpu().numpy())
        probs = np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)
        return Predictions(labels=(probs > 0.5).astype(int), probs=probs)
