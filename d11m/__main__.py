import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from . import tokenizer
from .checkpoint import load_model, save_model
from .dataset import Dataset
from .generation import generate
from .model import Model
from .training import train

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / 'model.pt'

CONTEXT_SIZE = 4
BATCH_SIZE = 32
EMBEDDING_DIMENSION = 64
EPOCHS = 100
NUMBER_OF_LAYERS = 2


def _get_input_from_argv(argv: list[str]) -> str:
    argv_len = len(argv)

    if argv_len < 2:
        print("Error: missing text argument.", file=sys.stderr)
        sys.exit(1)

    if argv_len > 2:
        print("Error: expected a single argument.", file=sys.stderr)
        sys.exit(1)

    return argv[1]


def _create_data_loader(tokens: list[int]) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = Dataset(
        torch.tensor(tokens, dtype=torch.long),
        context_size=CONTEXT_SIZE,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )


def _generate_output(model: Model) -> None:
    generated = generate(
        model,
        prompt="Ala ",
        max_new_tokens=100
    )

    print('Generated:', generated)


def main(argv: list[str]) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if CHECKPOINT_PATH.exists():
        model = load_model(CHECKPOINT_PATH, device)
        print(f'Loaded model: {CHECKPOINT_PATH}')
        _generate_output(model)
        return

    input_text = _get_input_from_argv(argv)

    tokens = [
        tokenizer.BOS,
        *tokenizer.encode(input_text),
        tokenizer.EOS,
    ]

    data_loader = _create_data_loader(tokens)

    model = Model(
        vocab_size=tokenizer.VOCAB_SIZE,
        context_size=CONTEXT_SIZE,
        embedding_dim=EMBEDDING_DIMENSION,
        number_of_layers=NUMBER_OF_LAYERS,
    ).to(device)

    train(model, data_loader, epochs=EPOCHS)
    save_model(model, CHECKPOINT_PATH)

    print(f'Saved model: {CHECKPOINT_PATH}')

    _generate_output(model)


if __name__ == '__main__':
    main(sys.argv)
