import json
from pathlib import Path

import torch

from .model import Model
from .tokenizer import ByteBPETokenizer


def save_model(model: Model, tokenizer: ByteBPETokenizer, path: Path) -> None:
    torch.save(
        {
            'config': {
                'context_size': model.context_size,
                'embedding_dim': model.token_embedding.embedding_dim,
                'number_of_layers': len(model.transformers),
            },
            'state_dict': model.state_dict(),
            'tokenizer': _get_tokenizer_merge_list(tokenizer),
        },
        path,
    )


def save_tokenizer(tokenizer: ByteBPETokenizer, path: Path) -> None:
    merges = _get_tokenizer_merge_list(tokenizer)
    path.write_text(json.dumps(merges, indent=2), encoding='utf-8')


def load_model(path: Path, device: torch.device) -> tuple[Model, ByteBPETokenizer]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint['config']

    tokenizer = _create_tokenizer(checkpoint['tokenizer'])
    model = Model(vocab_size=tokenizer.vocab_size, **config).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    return model, tokenizer


def load_tokenizer(path: Path) -> ByteBPETokenizer:
    with path.open(encoding='utf-8') as f:
        data = json.load(f)

    return _create_tokenizer(data)


def _create_tokenizer(merges: list[list[int]]) -> ByteBPETokenizer:
    tokenizer = ByteBPETokenizer()

    for rank, pair in enumerate(merges):
        left, right = pair
        key = (left, right)
        token_id = ByteBPETokenizer.FIRST_MERGE_TOKEN + rank
        tokenizer.merges[key] = token_id
        tokenizer.vocab[token_id] = tokenizer.vocab[left] + tokenizer.vocab[right]

    return tokenizer


def _get_tokenizer_merge_list(tokenizer: ByteBPETokenizer) -> list[list[int]]:
    return [list(pair) for pair in sorted(tokenizer.merges, key=tokenizer.merges.__getitem__)]
