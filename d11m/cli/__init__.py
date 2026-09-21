from argparse import ArgumentParser

from . import generate, train

__all__ = ['main']


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(description='Train a small language model or generate text.')

    commands = parser.add_subparsers(dest='command', required=True)

    train.configure_parser(commands.add_parser('train', help='Train a new model and save it.'))
    generate.configure_parser(commands.add_parser('generate', help='Generate text from a saved model.'))

    args = parser.parse_args(argv)
    args.handler(args, args.command_parser)
