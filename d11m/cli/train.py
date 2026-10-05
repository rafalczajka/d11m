from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..checkpoint import load_checkpoint, load_tokenizer, save_checkpoint
from ..data import DATASETS
from ..dataset import Dataset
from ..model import Model
from ..tokenizer import ByteBPETokenizer
from ..training import create_optimizer, train_gen
from ._common import CHECKPOINT_PATH, get_device, load_texts
from ._validation import positive_int, validate_checkpoint_path, validate_tokenizer_path

DEFAULT_TOKENIZER_FILE = 'tokenizer.json'


def configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--max-samples', type=positive_int)
    parser.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER_FILE)
    parser.add_argument('--context-size', type=positive_int, default=128)
    parser.add_argument('--batch-size', type=positive_int, default=32)
    parser.add_argument('--embedding-dim', type=positive_int, default=128)
    parser.add_argument('--epochs', type=positive_int, default=10)
    parser.add_argument('--number-of-layers', type=positive_int, default=4)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT_PATH)
    parser.add_argument('--resume', action='store_true')
    parser.set_defaults(handler=run, command_parser=parser)


def run(args: Namespace, parser: ArgumentParser) -> None:
    _validate_args(args, parser)

    device = get_device()
    print(f'Device: {device}')

    if args.resume:
        model, tokenizer, optimizer = _load_training_components(
            checkpoint_path=args.checkpoint,
            device=device,
        )
    else:
        model, tokenizer, optimizer = _create_training_components(
            tokenizer_path=args.tokenizer,
            context_size=args.context_size,
            embedding_dim=args.embedding_dim,
            number_of_layers=args.number_of_layers,
            device=device,
        )

    texts = load_texts(parser, args.dataset, args.max_samples)
    tokens: list[int] = []

    for text in texts:
        tokens.extend(
            [
                tokenizer.BOS,
                *tokenizer.encode(text),
                tokenizer.EOS,
            ]
        )

    if len(tokens) <= model.context_size:
        parser.error('Training data with BOS and EOS must be longer than context size.')

    token_tensor = torch.tensor(tokens, dtype=torch.long)
    del tokens

    data_loader = _create_data_loader(
        tokens=token_tensor,
        context_size=model.context_size,
        batch_size=args.batch_size,
    )

    _train_and_show_progress(
        model=model,
        data_loader=data_loader,
        epochs=args.epochs,
        optimizer=optimizer,
    )

    save_checkpoint(model, tokenizer, optimizer=optimizer, path=args.checkpoint)
    print(f'Saved model: {args.checkpoint}')


def _validate_args(args: Namespace, parser: ArgumentParser) -> None:
    if args.resume:
        validate_checkpoint_path(args.checkpoint, parser)
    else:
        validate_tokenizer_path(args.tokenizer, parser)


def _create_training_components(
    tokenizer_path: Path,
    context_size: int,
    embedding_dim: int,
    number_of_layers: int,
    device: torch.device,
) -> tuple[Model, ByteBPETokenizer, Optimizer]:
    tokenizer = load_tokenizer(tokenizer_path)

    model = Model(
        vocab_size=tokenizer.vocab_size,
        context_size=context_size,
        embedding_dim=embedding_dim,
        number_of_layers=number_of_layers,
    ).to(device)

    optimizer = create_optimizer(model)
    return model, tokenizer, optimizer


def _load_training_components(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[Model, ByteBPETokenizer, Optimizer]:
    model, tokenizer, optimizer_state = load_checkpoint(checkpoint_path, device)
    optimizer = create_optimizer(model, optimizer_state)
    return model, tokenizer, optimizer


def _create_data_loader(
    tokens: Tensor,
    context_size: int,
    batch_size: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = Dataset(
        tokens,
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
    optimizer: Optimizer,
) -> None:
    results = train_gen(
        model,
        data_loader,
        epochs=epochs,
        optimizer=optimizer,
    )

    with tqdm(total=epochs * len(data_loader), unit='batch') as progress:
        for epoch, average_loss in results:
            progress.set_description(f'Epoch {epoch}/{epochs}', refresh=False)
            progress.set_postfix(loss=f'{average_loss:.4f}', refresh=False)
            progress.update(1)
