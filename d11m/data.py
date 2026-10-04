from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice

from datasets import load_dataset


@dataclass(frozen=True)
class DatasetSource:
    path: str
    config: str | None
    split: str
    text_field: str


DATASETS: dict[str, DatasetSource] = {
    'wikipedia-pl': DatasetSource(
        path='wikimedia/wikipedia',
        config='20231101.pl',
        split='train',
        text_field='text',
    ),
}


def load_texts(
    name: str,
    split: str | None = None,
    max_samples: int | None = None,
) -> Iterator[str]:
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset: {name}. Available datasets: {", ".join(DATASETS)}.')

    if max_samples is not None and max_samples <= 0:
        raise ValueError('max_samples must be positive.')

    source = DATASETS[name]

    dataset = load_dataset(
        source.path,
        name=source.config,
        split=source.split if split is None else split,
    )

    for record in islice(dataset, max_samples):
        text = record.get(source.text_field)

        if not isinstance(text, str):
            raise ValueError(f'Dataset {name} must contain a string field: {source.text_field}.')

        if text.strip():
            yield text
