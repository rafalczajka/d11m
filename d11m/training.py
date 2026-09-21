from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from . import tokenizer
from .model import Model

DEFAULT_LEARNING_RATE = 0.001


def train(
    model: Model,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    epochs: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> float:
    last_loss = 0.0

    for _, average_loss in train_gen(model, data_loader, epochs, learning_rate):
        last_loss = average_loss

    return last_loss


def train_gen(
    model: Model,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    epochs: int,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> Iterator[tuple[int, float]]:
    model.train()

    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0

        for input_tokens, target_tokens in data_loader:
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            optimizer.zero_grad()

            logits = model(input_tokens)

            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.VOCAB_SIZE),
                target_tokens.reshape(-1),
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            batch_tokens = target_tokens.numel()

            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            average_loss = total_loss / total_tokens

            yield epoch + 1, average_loss
