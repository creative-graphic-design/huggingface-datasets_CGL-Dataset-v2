import os

import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "CGL-Dataset-v2.py"


@pytest.fixture
def data_dir() -> str:
    return "RADM_dataset.tar.gz"


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames="decode_rle",
    argvalues=(
        True,
        False,
    ),
)
@pytest.mark.parametrize(
    argnames="include_text_features",
    argvalues=(
        True,
        False,
    ),
)
def test_load_dataset(
    dataset_path: str,
    data_dir: str,
    include_text_features: bool,
    decode_rle: bool,
    expected_num_train: int = 60548,
    expected_num_test: int = 1035,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        data_dir=data_dir,
        decode_rle=decode_rle,
        include_text_features=include_text_features,
    )
    assert dataset["train"].num_rows == expected_num_train
    assert dataset["test"].num_rows == expected_num_test
