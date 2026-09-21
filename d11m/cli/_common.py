from argparse import ArgumentTypeError
from pathlib import Path

import torch

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / 'model.pt'


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def positive_int(value: str) -> int:
    number = int(value)

    if number <= 0:
        raise ArgumentTypeError('must be greater than zero')

    return number
