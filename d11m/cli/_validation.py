from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path


def positive_int(value: str) -> int:
    number = int(value)

    if number <= 0:
        raise ArgumentTypeError('must be greater than zero')

    return number


def validate_tokenizer_path(path: Path, parser: ArgumentParser) -> None:
    if not path.is_file():
        parser.error(f'Tokenizer not found: {path}. Run tokenizer train first.')
