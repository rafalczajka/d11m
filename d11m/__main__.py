import sys

from .dataset import Dataset
from .tokenizer import decode, encode


def _get_input_from_argv(argv: list[str]) -> str:
    argv_len = len(argv)

    if argv_len < 2:
        print("Error: missing text argument.", file=sys.stderr)
        sys.exit(1)

    if argv_len > 2:
        print("Error: expected a single argument.", file=sys.stderr)
        sys.exit(1)

    return argv[1]


if __name__ == '__main__':
    input_text = _get_input_from_argv(sys.argv)

    tokens= encode(input_text)
    decoded_tokens = decode(tokens)

    print('original:', input_text)
    print('encoded: ', tokens)
    print('decoded: ', decoded_tokens)

    dataset = Dataset(tokens, context_size=4)

    for input_tokens, target_tokens in dataset:
        print(input_tokens)
        print(target_tokens)
