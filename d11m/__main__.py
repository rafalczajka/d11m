import sys

import torch
from torch.utils.data import DataLoader

from d11m import tokenizer
from d11m.dataset import Dataset
from d11m.model import Model


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

    context_size = 4
    embedding_dim = 64
    batch_size = 32

    model = Model(
        vocab_size=tokenizer.VOCAB_SIZE,
        context_size=context_size,
        embedding_dim=embedding_dim,
    )

    dataset = Dataset(
        torch.tensor(tokens, dtype=torch.long),
        context_size=context_size
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    for input_tokens, target_tokens in data_loader:
        output = model(input_tokens)
