from torch import Tensor, arange, nn

from .transformer import Transformer


class Model(nn.Module):
    def __init__(self, vocab_size: int, context_size: int, embedding_dim: int) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_size, embedding_dim)

        self.transformer = Transformer(embedding_dim)

    def forward(self, input_tokens: Tensor) -> Tensor:
        embeddings = self._create_embeddings(input_tokens)
        embeddings = embeddings + self.attention(self.attention_norm(embeddings))
        embeddings = self.transformer(embeddings)

    def _create_embeddings(self, input_tokens: Tensor) -> Tensor:
        token_embeddings = self.token_embedding(input_tokens)

        positions = arange(input_tokens.shape[1], device=input_tokens.device)
        position_embeddings = self.position_embedding(positions)

        return token_embeddings + position_embeddings
