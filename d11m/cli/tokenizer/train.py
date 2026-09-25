from argparse import ArgumentParser, Namespace
from pathlib import Path

from ...checkpoint import save_tokenizer
from ...tokenizer import ByteBPETokenizer
from .._validation import positive_int


def configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument('text', help='Training text.')
    parser.add_argument('--vocab-size', type=positive_int, default=4096)
    parser.add_argument('--min-frequency', type=positive_int, default=2)
    parser.add_argument('--output', type=Path, default=Path('tokenizer.json'))
    parser.set_defaults(handler=run, command_parser=parser)


def _validate_args(args: Namespace, parser: ArgumentParser) -> None:
    if args.vocab_size < ByteBPETokenizer.FIRST_MERGE_TOKEN:
        parser.error(f'--vocab-size must be at least {ByteBPETokenizer.FIRST_MERGE_TOKEN}.')


def run(args: Namespace, parser: ArgumentParser) -> None:
    _validate_args(args, parser)

    tokenizer = ByteBPETokenizer()

    tokenizer.train(args.text, args.vocab_size, args.min_frequency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, args.output)

    print(f'Saved tokenizer: {args.output} (vocabulary size: {tokenizer.vocab_size})')
