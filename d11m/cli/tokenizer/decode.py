from argparse import ArgumentParser, Namespace
from pathlib import Path

from ...checkpoint import load_tokenizer
from .._validation import validate_tokenizer_path


def configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument('tokens', type=int, nargs='+')
    parser.add_argument('--tokenizer', type=Path, default=Path('tokenizer.json'))
    parser.set_defaults(handler=run, command_parser=parser)


def run(args: Namespace, parser: ArgumentParser) -> None:
    _validate_args(args, parser)

    tokenizer = load_tokenizer(args.tokenizer)
    text = tokenizer.decode(args.tokens)

    print(text)


def _validate_args(args: Namespace, parser: ArgumentParser) -> None:
    validate_tokenizer_path(args.tokenizer, parser)
