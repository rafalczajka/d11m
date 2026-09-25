from pathlib import Path

import torch

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / 'model.pt'


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
