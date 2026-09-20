ENCODING = 'utf-8'


def encode(text: str) -> list[int]:
    return list(text.encode(ENCODING))

def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode(ENCODING, errors='replace')
