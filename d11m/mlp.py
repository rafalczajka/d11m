from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        hidden_dim = embedding_dim * 4

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.network(embeddings)
