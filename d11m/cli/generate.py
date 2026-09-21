from argparse import ArgumentParser, Namespace
from pathlib import Path

from ..checkpoint import load_model
from ..generation import generate
from ._common import CHECKPOINT_PATH, get_device, positive_int


def configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument('prompt')
    parser.add_argument('--max-new-tokens', type=positive_int, default=100)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH)
    parser.set_defaults(handler=run, command_parser=parser)


def run(args: Namespace, parser: ArgumentParser) -> None:
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}. Run train first.')

    device = get_device()
    print(f'Device: {device}')

    model = load_model(args.checkpoint, device)

    generated = generate(model, prompt=args.prompt, max_new_tokens=args.max_new_tokens)

    print('Generated:', generated)
