from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..checkpoint import load_tokenizer, save_model
from ..dataset import Dataset
from ..model import Model
from ..training import train_gen
from ._common import CHECKPOINT_PATH, get_device, positive_int

DEFAULT_TOKENIZER_FILE = 'tokenizer.json'


def _create_data_loader(
    tokens: list[int],
    context_size: int,
    batch_size: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = Dataset(
        torch.tensor(tokens, dtype=torch.long),
        context_size=context_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def _train_and_show_progress(
    model: Model,
    data_loader: DataLoader[tuple[Tensor, Tensor]],
    epochs: int,
) -> None:
    with tqdm(total=epochs * len(data_loader), unit='batch') as progress:
        for epoch, average_loss in train_gen(model, data_loader, epochs=epochs):
            progress.set_description(f'Epoch {epoch}/{epochs}', refresh=False)
            progress.set_postfix(loss=f'{average_loss:.4f}', refresh=False)
            progress.update(1)


def configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument('text', help='Training text.')
    parser.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER_FILE)
    parser.add_argument('--context-size', type=positive_int, default=128)
    parser.add_argument('--batch-size', type=positive_int, default=32)
    parser.add_argument('--embedding-dim', type=positive_int, default=128)
    parser.add_argument('--epochs', type=positive_int, default=10)
    parser.add_argument('--number-of-layers', type=positive_int, default=4)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH)
    parser.set_defaults(handler=run, command_parser=parser)


def run(args: Namespace, parser: ArgumentParser) -> None:
    if not args.tokenizer.is_file():
        parser.error(f'Tokenizer not found: {args.tokenizer}. Run tokenizer train first.')

    device = get_device()
    print(f'Device: {device}')

    tokenizer = load_tokenizer(args.tokenizer)

    tokens = [
        tokenizer.BOS,
        *tokenizer.encode(args.text),
        tokenizer.EOS
    ]

    if len(tokens) <= args.context_size:
        parser.error('Training text with BOS and EOS must be longer than --context-size.')

    data_loader = _create_data_loader(tokens, args.context_size, args.batch_size)

    model = Model(
        vocab_size=tokenizer.vocab_size,
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        number_of_layers=args.number_of_layers,
    ).to(device)

    _train_and_show_progress(model, data_loader, epochs=args.epochs)

    save_model(model, tokenizer, args.checkpoint)
    print(f'Saved model: {args.checkpoint}')
