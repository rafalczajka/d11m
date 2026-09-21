from pathlib import Path

import torch

from .model import Model


def save_model(model: Model, path: Path) -> None:
    torch.save(
        {
            'config': {
                'vocab_size': model.token_embedding.num_embeddings,
                'context_size': model.context_size,
                'embedding_dim': model.token_embedding.embedding_dim,
                'number_of_layers': len(model.transformers),
            },
            'state_dict': model.state_dict(),
        },
        path,
    )


def load_model(path: Path, device: torch.device) -> Model:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = Model(**checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model
