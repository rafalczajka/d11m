from argparse import ArgumentParser
from collections.abc import Iterator
from pathlib import Path

import torch
from tqdm.auto import tqdm

from ..data import load_texts as load_data_texts

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / 'model.pt'


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_texts(
    parser: ArgumentParser,
    dataset: str,
    max_samples: int | None,
) -> Iterator[str]:
    has_text = False

    try:
        texts = load_data_texts(
            dataset,
            max_samples=max_samples,
        )

        for text in tqdm(
            texts,
            total=max_samples,
            desc='Loading data',
            unit='sample',
        ):
            has_text = True
            yield text
    except ValueError as error:
        parser.error(str(error))

    if not has_text:
        parser.error('Training data is empty.')
