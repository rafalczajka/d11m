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
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}

        tokens = list(text.encode('utf-8'))

        next_token_id = self.FIRST_MERGE_TOKEN

        while next_token_id < vocab_size:
            pair_counts = Counter(itertools.pairwise(tokens))

            if not pair_counts:
                break

            best_pair, frequency = pair_counts.most_common(1)[0]

            if frequency < min_frequency:
                break

            self.merges[best_pair] = next_token_id

            left_token, right_token = best_pair

            self.vocab[next_token_id] = (self.vocab[left_token] + self.vocab[right_token])

            tokens = self._merge_pair(
                tokens=tokens,
                pair=best_pair,
                new_token=next_token_id,
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

    def _get_best_pair(self, tokens: list[int]) -> tuple[int, int] | None:
        best_pair = None
        best_token_id = None

        for pair in itertools.pairwise(tokens):
            token_id = self.merges.get(pair)

            if token_id is None:
                continue

            if best_token_id is None or token_id < best_token_id:
                best_pair = pair
                best_token_id = token_id

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
