from argparse import ArgumentParser, Namespace
from pathlib import Path

from ...checkpoint import load_tokenizer
from .._validation import validate_tokenizer_path


def configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument('text')
    parser.add_argument('--tokenizer', type=Path, default=Path('tokenizer.json'))
    parser.set_defaults(handler=run, command_parser=parser)


def run(args: Namespace, parser: ArgumentParser) -> None:
    _validate_args(args, parser)

    tokenizer = load_tokenizer(args.tokenizer)
    tokens = tokenizer.encode(args.text)

    print('Tokens:', tokens)

    for token in tokens:
        print(f'{token}: {tokenizer.vocab[token]!r}')


def _validate_args(args: Namespace, parser: ArgumentParser) -> None:
    validate_tokenizer_path(args.tokenizer, parser)
