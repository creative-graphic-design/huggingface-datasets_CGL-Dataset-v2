import os

import datasets as ds
import pytest


@pytest.fixture
def org_name() -> str:
    return "creative-graphic-design"


@pytest.fixture
def dataset_name() -> str:
    return "CGL-Dataset-v2"


@pytest.fixture
def dataset_path(dataset_name: str) -> str:
    return f"{dataset_name}.py"


@pytest.fixture
def repo_id(org_name: str, dataset_name: str) -> str:
    return f"{org_name}/{dataset_name}"


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
@pytest.mark.parametrize(
    argnames="rename_category_names",
    argvalues=(
        True,
        False,
    ),
)
@pytest.mark.parametrize(
    argnames="subset_name",
    argvalues=(
        "default",
        "ralf",
    ),
)
def test_load_dataset(
    dataset_path: str,
    subset_name: str,
    include_text_features: bool,
    rename_category_names: bool,
    decode_rle: bool,
    expected_num_train: int = 60548,
    expected_num_test: int = 1035,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=subset_name,
        decode_rle=decode_rle,
        include_text_features=include_text_features,
        rename_category_names=rename_category_names,
        token=True,
    )
    assert isinstance(dataset, ds.DatasetDict)
    assert dataset["train"].num_rows == expected_num_train
    assert dataset["test"].num_rows == expected_num_test


@pytest.mark.skipif(
    condition=bool(os.environ.get("CI", False)),
    reason=(
        "Because this loading script downloads a large dataset, "
        "we will skip running it on CI."
    ),
)
@pytest.mark.parametrize(
    argnames="subset_name",
    argvalues=(
        # "default",
        "ralf",
    ),
)
def test_push_to_hub(
    repo_id: str,
    subset_name: str,
    dataset_path: str,
):
    dataset = ds.load_dataset(
        path=dataset_path,
        name=subset_name,
        decode_rle=False,
        include_text_features=True,
        rename_category_names=True,
        token=True,
    )
    assert isinstance(dataset, ds.DatasetDict)

    def split_dataset(dataset: ds.DatasetDict):
        #
        # Rename `test` (with no annotation) to `no_annotation`
        #
        no_annotation_dataset = dataset["test"]

        #
        # Split the dataset into train:valid:test = 8:1:1
        #
        # First, split train into train and test at 8:2
        tng_tst_dataset = dataset["train"].train_test_split(test_size=0.2)
        # Then, split test into valid and test at 1:1 ratio to make train:valid:test = 8:1:1
        val_tst_dataset = tng_tst_dataset["test"].train_test_split(test_size=0.5)

        # Reorganize the split dataset
        tng_dataset = tng_tst_dataset["train"]
        val_dataset = val_tst_dataset["train"]
        tst_dataset = val_tst_dataset["test"]

        dataset = ds.DatasetDict(
            train=tng_dataset,
            validation=val_dataset,
            test=tst_dataset,
            no_annotation=no_annotation_dataset,
        )

        # Check if the split is correct
        assert (
            tng_dataset.num_rows == 48438
            and val_dataset.num_rows == 6055
            and tst_dataset.num_rows == 6055
            and no_annotation_dataset.num_rows == 1035
        ), dataset

        return dataset

    if subset_name == "ralf":
        dataset = split_dataset(dataset)

    #
    # Push the dataset to the huggingface hub
    #
    dataset.push_to_hub(repo_id=repo_id, config_name=subset_name, private=True)
