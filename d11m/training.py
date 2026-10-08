from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from .model import Model

LEARNING_RATE = 0.001


def create_optimizer(
    model: Model,
    optimizer_state: dict[str, Any] | None = None,
    learning_rate: float = LEARNING_RATE,
) -> Optimizer:
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    return optimizer


def train(
    model: Model,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    epochs: int,
    optimizer: Optimizer,
) -> float:
    last_loss = 0.0

    for _, average_loss in train_gen(model, data_loader, epochs, optimizer):
        last_loss = average_loss

    return last_loss


def train_gen(
    model: Model,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    epochs: int,
    optimizer: Optimizer,
) -> Iterator[tuple[int, float]]:
    model.train()

    device = next(model.parameters()).device

    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0

        for input_tokens, target_tokens in data_loader:
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)

            optimizer.zero_grad()

            loss = _calculate_loss(model, input_tokens, target_tokens)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_tokens = target_tokens.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            average_loss = total_loss / total_tokens

            yield epoch + 1, average_loss


def evaluate(
    model: Model,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
) -> float:
    was_training = model.training
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    try:
        model.eval()

        with torch.inference_mode():
            for input_tokens, target_tokens in data_loader:
                input_tokens = input_tokens.to(device)
                target_tokens = target_tokens.to(device)

                loss = _calculate_loss(model, input_tokens, target_tokens)

                batch_tokens = target_tokens.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        if total_tokens == 0:
            raise ValueError('Validation data contains no target tokens.')

        return total_loss / total_tokens
    finally:
        model.train(was_training)


def _calculate_loss(
    model: Model,
    input_tokens: Tensor,
    target_tokens: Tensor,
) -> Tensor:
    logits = model(input_tokens)

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_tokens.reshape(-1),
    )
