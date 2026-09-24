from collections.abc import Generator

import torch
from torch import Tensor

from .model import Model
from .tokenizer import ByteBPETokenizer

SOFTMAX_TEMPERATURE = 0.8


def _get_next_token(
    logits: Tensor,
    temperature: float,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0")

    probabilities = torch.softmax(
        logits / temperature,
        dim=-1
    )

    return torch.multinomial(
        probabilities,
        num_samples=1
    ).squeeze(0)


def generate(
    model: Model,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    tokens = list(generate_gen(model, tokenizer, prompt, max_new_tokens))
    return tokenizer.decode(tokens)


def generate_gen(
    model: Model,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_new_tokens: int,
) -> Generator[int, None, None]:
    model.eval()

    ignored_tokens = [
        tokenizer.PAD,
        tokenizer.BOS,
        tokenizer.QUESTION,
        tokenizer.ANSWER,
    ]

    tokens = torch.tensor(
        [
            tokenizer.BOS,
            *tokenizer.encode(prompt),
        ],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )

    for _ in range(max_new_tokens):
        with torch.no_grad():
            input_tokens = tokens[-model.context_size:]
            input_tokens = input_tokens.unsqueeze(0)

            logits = model(input_tokens)

            last_token_logits = logits[0, -1]
            last_token_logits[ignored_tokens] = float('-inf')

            next_token = _get_next_token(last_token_logits, SOFTMAX_TEMPERATURE)

            token_id = int(next_token.item())

            if token_id == tokenizer.EOS:
                break

            tokens = torch.cat([
                tokens,
                next_token.unsqueeze(0)
            ])

        yield token_id
