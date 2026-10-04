from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm.auto import tqdm

from ...checkpoint import save_tokenizer
from ...data import DATASETS
from ...tokenizer import ByteBPETokenizer
from .._common import load_texts
from .._validation import positive_int


def configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--max-samples', type=positive_int)
    parser.add_argument('--vocab-size', type=positive_int, default=4096)
    parser.add_argument('--min-frequency', type=positive_int, default=2)
    parser.add_argument('--output', type=Path, default=Path('tokenizer.json'))
    parser.set_defaults(handler=run, command_parser=parser)


def run(args: Namespace, parser: ArgumentParser) -> None:
    _validate_args(args, parser)

    tokenizer = ByteBPETokenizer()

    texts = load_texts(parser, args.dataset, args.max_samples)
    total_merges = args.vocab_size - tokenizer.FIRST_MERGE_TOKEN

    with tqdm(total=total_merges, desc='Training tokenizer', unit='merge') as progress:
        for _ in tokenizer.train_gen(texts, args.vocab_size, args.min_frequency):
            progress.update(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, args.output)

    print(f'Saved tokenizer: {args.output} (vocabulary size: {tokenizer.vocab_size})')


def _validate_args(args: Namespace, parser: ArgumentParser) -> None:
    if args.vocab_size < ByteBPETokenizer.FIRST_MERGE_TOKEN:
        parser.error(f'--vocab-size must be at least {ByteBPETokenizer.FIRST_MERGE_TOKEN}.')
