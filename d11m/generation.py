import torch

from . import tokenizer
from .model import Model


def generate(
    model: Model,
    prompt: str,
    max_new_tokens: int,
) -> str:
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

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tokens = tokens[-model.context_size:]
            input_tokens = input_tokens.unsqueeze(0)

            logits = model(input_tokens)
            last_token_logits = logits[0, -1]
            last_token_logits[ignored_tokens] = float('-inf')
            next_token = torch.argmax(last_token_logits) # TODO: use softmax

            if next_token.item() == tokenizer.EOS:
                break

            tokens = torch.cat([
                tokens,
                next_token.unsqueeze(0)
            ])

    return tokenizer.decode(tokens.tolist())
