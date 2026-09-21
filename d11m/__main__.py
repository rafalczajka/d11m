import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import tokenizer
from .dataset import Dataset
from .model import Model

CONTEXT_SIZE = 4
BATCH_SIZE = 32
EMBEDDING_DIMENSION = 64
EPOCHS = 100
NUMBER_OF_LAYERS = 2


def _get_input_from_argv(argv: list[str]) -> str:
    argv_len = len(argv)

    if argv_len < 2:
        print("Error: missing text argument.", file=sys.stderr)
        sys.exit(1)

    if argv_len > 2:
        print("Error: expected a single argument.", file=sys.stderr)
        sys.exit(1)

    return argv[1]


def generate(
    model: Model,
    prompt: str,
    max_new_tokens: int,
) -> str:
    model.eval()

    tokens = torch.tensor(
        [
            tokenizer.BOS,
            *tokenizer.encode(prompt),
        ],
        dtype=torch.long
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tokens = tokens[-model.context_size:]
            input_tokens = input_tokens.unsqueeze(0)

            logits = model(input_tokens)

            last_token_logits = logits[0, -1]

            next_token = torch.argmax(last_token_logits)

            if next_token.item() == tokenizer.EOS:
                break

            tokens = torch.cat([
                tokens,
                next_token.unsqueeze(0)
            ])

    return tokenizer.decode(tokens.tolist())


if __name__ == '__main__':
    input_text = _get_input_from_argv(sys.argv)

    tokens = [
        tokenizer.BOS,
        *tokenizer.encode(input_text),
        tokenizer.EOS,
    ]

    dataset = Dataset(
        torch.tensor(tokens, dtype=torch.long),
        context_size=CONTEXT_SIZE,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = Model(
        vocab_size=tokenizer.VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        embedding_dim=EMBEDDING_DIMENSION,
        number_of_layers=NUMBER_OF_LAYERS,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for input_tokens, target_tokens in data_loader:
            optimizer.zero_grad()

            logits = model(input_tokens)

            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.VOCAB_SIZE),
                target_tokens.reshape(-1),
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"loss: {average_loss:.4f}"
        )

    generated = generate(
        model,
        prompt="Ala ",
        max_new_tokens=100
    )

    print(generated)
