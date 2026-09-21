from torch import Tensor, nn

from .mlp import MLP
from .self_attention import SelfAttention


class Transformer(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = SelfAttention(embedding_dim)

        self.mlp_norm = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim)

    def forward(self, embeddings: Tensor) -> Tensor:
        embeddings = embeddings + self.attention(self.attention_norm(embeddings))
        hidden_states = embeddings + self.mlp(self.mlp_norm(embeddings))
        return hidden_states
