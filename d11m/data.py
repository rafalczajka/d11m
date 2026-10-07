import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

from datasets import disable_progress_bars, load_dataset
from datasets.utils import logging as datasets_logging
from huggingface_hub import logging as hub_logging
from huggingface_hub.utils import disable_progress_bars as disable_hub_progress_bars


@dataclass(frozen=True)
class DatasetSource:
    path: str
    config: str | None
    hf_split: str
    text_field: str
    id_field: str
    default_split: str = 'train'


DATASETS: dict[str, DatasetSource] = {
    'wikipedia-pl': DatasetSource(
        path='wikimedia/wikipedia',
        config='20231101.pl',
        hf_split='train',
        text_field='text',
        id_field='id',
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
    requested_split = source.default_split if split is None else split

    if requested_split not in ('train', 'validation', 'test'):
        raise ValueError(f'Unknown split: {requested_split}. Available splits: train, validation, test.')

    dataset = load_dataset(
        source.path,
        name=source.config,
        split=source.hf_split,
    )

    samples = 0

    for record in dataset:
        record_id = record.get(source.id_field)

        if not isinstance(record_id, (str, int)) or record_id == '':
            raise ValueError(f'Dataset {name} must contain a stable string or integer ID field: {source.id_field}.')

        if _get_split(str(record_id)) != requested_split:
            continue

        text = record.get(source.text_field)

        if not isinstance(text, str):
            raise ValueError(f'Dataset {name} must contain a string field: {source.text_field}.')

        if text.strip():
            yield text
            samples += 1

            if max_samples is not None and samples >= max_samples:
                return


def configure_huggingface_output() -> None:
    datasets_logging.set_verbosity_error()
    hub_logging.set_verbosity_error()

    disable_progress_bars()
    disable_hub_progress_bars()


def _get_split(record_id: str) -> str:
    digest = hashlib.sha256(record_id.encode('utf-8')).digest()
    bucket = int.from_bytes(digest[:4], byteorder='big') % 100

    if bucket < 90:
        return 'train'

    if bucket < 95:
        return 'validation'

    return 'test'
