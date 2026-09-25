from argparse import ArgumentParser

from . import decode, encode, train

__all__ = ['configure_parser']


def configure_parser(parser: ArgumentParser) -> None:
    commands = parser.add_subparsers(dest='tokenizer_command', required=True)
    train.configure_parser(commands.add_parser('train', help='Train a BPE tokenizer.'))
    encode.configure_parser(commands.add_parser('encode', help='Show token IDs and bytes.'))
    decode.configure_parser(commands.add_parser('decode', help='Decode token IDs to text.'))
