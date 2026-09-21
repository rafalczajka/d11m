import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from . import tokenizer
from .model import Model


def train(
    model: Model,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    epochs: int,
    learning_rate: float = 0.001,
) -> None:
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
            optimizer.step()
            batch_tokens = target_tokens.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        average_loss = total_loss / total_tokens

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"loss: {average_loss:.4f}"
        )
