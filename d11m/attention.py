import torch.nn.functional as F
from torch import Tensor, nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, embeddings: Tensor) -> Tensor:
        queries = self.query(embeddings)
        keys = self.key(embeddings)
        values = self.value(embeddings)

        return F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            is_causal=True,
        )
