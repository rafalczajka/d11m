from argparse import ArgumentParser, Namespace
from pathlib import Path

from ..checkpoint import load_model
from ..generation import generate_gen
from ._common import CHECKPOINT_PATH, get_device
from ._validation import positive_int


def configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument('prompt')
    parser.add_argument('--max-new-tokens', type=positive_int, default=100)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH)
    parser.set_defaults(handler=run, command_parser=parser)


def run(args: Namespace, parser: ArgumentParser) -> None:
    _validate_args(args, parser)

    device = get_device()
    print(f'Device: {device}')

    model, tokenizer = load_model(args.checkpoint, device)

    print('Generated:', args.prompt, end='', flush=True)
    tokens = generate_gen(model, tokenizer, args.prompt, args.max_new_tokens)

    for text in tokenizer.decode_stream(tokens):
        print(text, end='', flush=True)

    print()


def _validate_args(args: Namespace, parser: ArgumentParser) -> None:
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}. Run train first.')
