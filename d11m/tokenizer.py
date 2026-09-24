import itertools
from collections import Counter


class ByteBPETokenizer:
    PAD = 256
    BOS = 257
    EOS = 258
    QUESTION = 259
    ANSWER = 260

    FIRST_MERGE_TOKEN = 261

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.merge_ranks: dict[tuple[int, int], int] = {}

        self.vocab: dict[int, bytes] = {
            token_id: bytes([token_id])
            for token_id in range(256)
        }

    @property
    def vocab_size(self) -> int:
        return self.FIRST_MERGE_TOKEN + len(self.merges)

    def train(
        self,
        text: str,
        vocab_size: int,
        min_frequency: int = 2,
    ) -> None:
        if vocab_size < self.FIRST_MERGE_TOKEN:
            raise ValueError(f"vocab_size must be at least {self.FIRST_MERGE_TOKEN}")

        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")

        self.merges.clear()
        self.merge_ranks.clear()
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}

        tokens = list(text.encode('utf-8'))

        next_token_id = self.FIRST_MERGE_TOKEN

        while next_token_id < vocab_size:
            pair_counts = self._count_pairs(tokens)

            if not pair_counts:
                break

            best_pair, frequency = pair_counts.most_common(1)[0]

            if frequency < min_frequency:
                break

            new_token_id = next_token_id

            self.merges[best_pair] = new_token_id
            self.merge_ranks[best_pair] = len(self.merge_ranks)

            left_token, right_token = best_pair

            self.vocab[new_token_id] = (self.vocab[left_token] + self.vocab[right_token])

            tokens = self._merge_pair(
                tokens=tokens,
                pair=best_pair,
                new_token=new_token_id,
            )

            next_token_id += 1

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode('utf-8'))

        while True:
            pair = self._get_best_pair(tokens)

            if pair is None:
                break

            tokens = self._merge_pair(
                tokens=tokens,
                pair=pair,
                new_token=self.merges[pair],
            )

        return tokens

    def decode(self, tokens: list[int]) -> str:
        data = bytearray()

        special_tokens = {
            self.PAD,
            self.BOS,
            self.EOS,
            self.QUESTION,
            self.ANSWER,
        }

        for token in tokens:
            if token in special_tokens:
                continue

            token_bytes = self.vocab.get(token)

            if token_bytes is None:
                raise ValueError(f"Unknown token: {token}")

            data.extend(token_bytes)

        return data.decode('utf-8', errors='replace')

    def _count_pairs(self, tokens: list[int]) -> Counter[tuple[int, int]]:
        return Counter(itertools.pairwise(tokens))

    def _get_best_pair(self, tokens: list[int]) -> tuple[int, int] | None:
        best_pair = None
        best_rank = None

        for pair in itertools.pairwise(tokens):
            rank = self.merge_ranks.get(pair)

            if rank is None:
                continue

            if best_rank is None or rank < best_rank:
                best_pair = pair
                best_rank = rank

        return best_pair

    def _merge_pair(
        self,
        tokens: list[int],
        pair: tuple[int, int],
        new_token: int,
    ) -> list[int]:
        result = []
        index = 0

        while index < len(tokens):
            if (
                index < len(tokens) - 1
                and tokens[index] == pair[0]
                and tokens[index + 1] == pair[1]
            ):
                result.append(new_token)
                index += 2
                continue

            result.append(tokens[index])
            index += 1

        return result
