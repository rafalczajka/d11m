import sys

import torch
from torch.utils.data import DataLoader

from . import tokenizer
from .dataset import Dataset
from .model import Model

CONTEXT_SIZE = 4
BATCH_SIZE = 32
EMBEDDING_DIMENSION = 64


def _get_input_from_argv(argv: list[str]) -> str:
    argv_len = len(argv)

    if argv_len < 2:
        print("Error: missing text argument.", file=sys.stderr)
        sys.exit(1)

    if argv_len > 2:
        print("Error: expected a single argument.", file=sys.stderr)
        sys.exit(1)

    return argv[1]


if __name__ == '__main__':
    input_text = _get_input_from_argv(sys.argv)
    tokens = tokenizer.encode(input_text)

    dataset = Dataset(
        torch.tensor(tokens, dtype=torch.long),
        context_size=CONTEXT_SIZE
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = Model(
        vocab_size=tokenizer.VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        embedding_dim=EMBEDDING_DIMENSION,
    )

    for input_tokens, target_tokens in data_loader:
        output = model(input_tokens)
