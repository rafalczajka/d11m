import sys

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

    encoded = encode(input_text)
    decoded = decode(encoded)

    print('original:', input_text)
    print('encoded: ', encoded)
    print('decoded: ', decoded)
