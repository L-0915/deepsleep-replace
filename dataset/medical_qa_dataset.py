"""Medical QA streaming dataset for DeepSleep continued pretraining.

Loads from local Malikeh1375/medical-question-answering-datasets parquet files.
Converts instruction/input/output format to plain text for language modeling.
Supports sequence packing (zero padding, full Flash Attention throughput).
"""

import os
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, concatenate_datasets


SUBSETS = [
    "chatdoctor_healthcaremagic",
    "chatdoctor_icliniq",
    "medical_meadow_cord19",
    "medical_meadow_health_advice",
    "medical_meadow_medical_flashcards",
    "medical_meadow_mediqa",
    "medical_meadow_medqa",
    "medical_meadow_mmmlu",
    "medical_meadow_pubmed_causal",
    "medical_meadow_wikidoc",
    "medical_meadow_wikidoc_patient_information",
]

DEFAULT_DATA_DIR = (
    "/public/huggingface-datasets/Malikeh1375/medical-question-answering-datasets"
)


def _format_sample(sample: dict) -> str:
    """Convert instruction/input/output to plain text."""
    instruction = sample.get("instruction", "").strip()
    inp = sample.get("input", "").strip()
    output = sample.get("output", "").strip()

    parts = []
    if instruction:
        parts.append(instruction)
    if inp:
        parts.append(inp)
    if output:
        parts.append(output)

    return "\n\n".join(parts)


def _load_all_texts(data_dir: str, subsets: list[str] | None = None) -> list[str]:
    """Load all texts from parquet files, deduplicate, and filter."""
    base = Path(data_dir)
    use_subsets = subsets or SUBSETS
    all_datasets = []

    for subset in use_subsets:
        subset_dir = base / subset
        if not subset_dir.exists():
            continue
        parquet_files = list(subset_dir.glob("*.parquet"))
        if not parquet_files:
            continue
        ds = load_dataset(
            "parquet",
            data_files=[str(f) for f in parquet_files],
            split="train",
        )
        all_datasets.append(ds)

    if not all_datasets:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    combined = concatenate_datasets(all_datasets)

    texts = []
    seen = set()
    for sample in combined:
        text = _format_sample(sample)
        text = text.strip()
        if len(text) < 30:
            continue
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)

    return texts


class MedicalQADataset(IterableDataset):
    """Streaming medical QA dataset with sequence packing for continued pretraining.

    Loads all text data into memory on init, then iterates with deterministic
    shuffle and on-the-fly tokenization + packing.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        data_dir: str = DEFAULT_DATA_DIR,
        seed: int = 42,
        num_samples: int | None = None,
        min_length: int = 30,
        pack_sequences: bool = True,
        subsets: list[str] | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.num_samples = num_samples
        self.min_length = min_length
        self.pack_sequences = pack_sequences

        self.bos_id = tokenizer.bos_token_id or tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
        self.pad_id = tokenizer.pad_token_id or 0

        print(f"Loading medical QA data from {data_dir} ...")
        self.texts = _load_all_texts(data_dir, subsets)
        print(f"  Loaded {len(self.texts)} unique texts")

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        if self.pack_sequences:
            est_tokens = sum(len(t) // 3 for t in self.texts)
            return max(1, est_tokens // self.max_length)
        return len(self.texts)

    def _tokenize_doc(self, text: str) -> list[int]:
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        return [self.bos_id] + tokens + [self.eos_id]

    def __iter__(self):
        rng = random.Random(self.seed)
        indices = list(range(len(self.texts)))
        rng.shuffle(indices)

        if self.pack_sequences:
            yield from self._iter_packed(indices)
        else:
            yield from self._iter_padded(indices)

    def _iter_packed(self, indices):
        yielded = 0
        buffer_ids: list[int] = []

        for idx in indices:
            text = self.texts[idx]
            if len(text) < self.min_length:
                continue

            tokens = self._tokenize_doc(text)
            buffer_ids.extend(tokens)

            while len(buffer_ids) >= self.max_length:
                chunk = buffer_ids[: self.max_length]
                buffer_ids = buffer_ids[self.max_length :]

                labels = list(chunk)
                for i in range(1, len(labels)):
                    if chunk[i] == self.bos_id:
                        labels[i] = -100

                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
                yielded += 1
                if self.num_samples is not None and yielded >= self.num_samples:
                    return

    def _iter_padded(self, indices):
        yielded = 0

        for idx in indices:
            text = self.texts[idx]
            if len(text) < self.min_length:
                continue

            tokens = self._tokenize_doc(text)
            seq_len = len(tokens)

            input_ids = tokens + [self.pad_id] * (self.max_length - seq_len)
            labels = tokens + [-100] * (self.max_length - seq_len)
            attention_mask = [1] * seq_len + [0] * (self.max_length - seq_len)

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
            yielded += 1
            if self.num_samples is not None and yielded >= self.num_samples:
                return
