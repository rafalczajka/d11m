import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from d11m import tokenizer

from .dataset import Dataset


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
    decoded_tokens = tokenizer.decode(tokens)

    print('original:', input_text)
    print('encoded: ', tokens)
    print('decoded: ', decoded_tokens)

    dataset = Dataset(
        torch.tensor(tokens, dtype=torch.long),
        context_size=4,
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    token_embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim=32)

    for input_tokens, target_tokens in dataloader:
        input_embeddings = token_embedding(input_tokens)

        print('input_tokens.shape:', input_tokens.shape)
        print('target_tokens.shape:', target_tokens.shape)
        print('input tokens: ', input_tokens[0])
        print('target tokens:', target_tokens[0])
        print('input_embeddings.shape:', input_embeddings.shape)
        break
