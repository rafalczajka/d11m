from torch import Tensor, arange, nn

from .transformer import Transformer


class Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embedding_dim: int,
        number_of_layers: int
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_size, embedding_dim)

        self.transformers = nn.ModuleList([
            Transformer(embedding_dim)
            for _ in range(number_of_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_tokens: Tensor) -> Tensor:
        embeddings = self._create_embeddings(input_tokens)

        for transformer in self.transformers:
            embeddings = transformer(embeddings)

        embeddings = self.final_norm(embeddings)
        return self.output_layer(embeddings)

    def _create_embeddings(self, input_tokens: Tensor) -> Tensor:
        token_embeddings = self.token_embedding(input_tokens)

        positions = arange(input_tokens.shape[1], device=input_tokens.device)
        position_embeddings = self.position_embedding(positions)

        return token_embeddings + position_embeddings
