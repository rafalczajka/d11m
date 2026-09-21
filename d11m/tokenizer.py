PAD = 256
BOS = 257
EOS = 258
QUESTION = 259
ANSWER = 260

VOCAB_SIZE = 261

_ENCODING = 'utf-8'


def encode(text: str) -> list[int]:
    return list(text.encode(_ENCODING))


def decode(tokens: list[int]) -> str:
    text_tokens = [token for token in tokens if token < PAD]
    return bytes(text_tokens).decode(_ENCODING, errors='replace')
