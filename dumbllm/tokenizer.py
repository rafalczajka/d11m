def encode(text):
    return list(text.encode("utf-8"))

def decode(tokens):
    tokens = [t for t in tokens if t < 256]
    return bytes(tokens).decode("utf-8", errors="replace")
