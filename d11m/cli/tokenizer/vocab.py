import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from ...checkpoint import load_tokenizer
from .._validation import validate_tokenizer_path


def configure_parser(parser: ArgumentParser) -> None:
    parser.add_argument('--tokenizer', type=Path, default=Path('tokenizer.json'))
    parser.add_argument('--output', type=Path, default=Path('vocab.json'))
    parser.set_defaults(handler=run, command_parser=parser)


def _validate_args(args: Namespace, parser: ArgumentParser) -> None:
    validate_tokenizer_path(args.tokenizer, parser)

    if args.output.resolve() == args.tokenizer.resolve():
        parser.error('Vocabulary output must not overwrite the tokenizer file.')


def run(args: Namespace, parser: ArgumentParser) -> None:
    _validate_args(args, parser)

    tokenizer = load_tokenizer(args.tokenizer)

    vocabulary = {
        token: data.decode('utf-8', errors='backslashreplace')
        for token, data in tokenizer.vocab.items()
    }

    vocabulary.update({
        tokenizer.PAD: '<PAD>',
        tokenizer.BOS: '<BOS>',
        tokenizer.EOS: '<EOS>',
        tokenizer.QUESTION: '<QUESTION>',
        tokenizer.ANSWER: '<ANSWER>',
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(
        json.dumps(dict(sorted(vocabulary.items())), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    print(f'Saved vocabulary: {args.output} ({len(vocabulary)} tokens)')
