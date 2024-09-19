# Copyright 2024 Shunsuke Kitada and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script was generated from shunk031/cookiecutter-huggingface-datasets.
#

from __future__ import annotations

import ast
import os
import pathlib
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import datasets as ds
import torch
from datasets.utils.logging import get_logger
from hfcocoapi.models import CategoryData, ImageData
from hfcocoapi.processors import InstancesProcessor
from hfcocoapi.processors.instances import InstanceExample
from hfcocoapi.tasks import InstancesAnnotationData
from hfcocoapi.typehint import (
    CategoryId,
    ImageId,
    JsonDict,
    PathLike,
)
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm
from typing_extensions import Self

if TYPE_CHECKING:
    from ralfpt.saliency_detection import SaliencyTester
    from simple_lama_inpainting import SimpleLama

logger = get_logger(__name__)


_DESCRIPTION = """\
CGL-Dataset V2 is a dataset for the task of automatic graphic layout design of advertising posters, containing 60,548 training samples and 1035 testing samples. It is an extension of CGL-Dataset.
"""

_CITATION = """\
@inproceedings{li2023relation,
    title={Relation-Aware Diffusion Model for Controllable Poster Layout Generation},
    author={Li, Fengheng and Liu, An and Feng, Wei and Zhu, Honghe and Li, Yaoyu and Zhang, Zheng and Lv, Jingjing and Zhu, Xin and Shen, Junjie and Lin, Zhangang},
    booktitle={Proceedings of the 32nd ACM international conference on information & knowledge management},
    pages={1249--1258},
    year={2023}
}
"""

_HOMEPAGE = "https://github.com/liuan0803/RADM"

_LICENSE = """\
Unknown
"""

_URLS = {
    "v2": "https://huggingface.co/datasets/shunk031-private/RADM-private/resolve/main/RADM_dataset.tar.gz",
    "v1": [
        "https://huggingface.co/datasets/shunk031-private/CGL-Dataset-private/resolve/main/layout_imgs_6w_1.zip",
        "https://huggingface.co/datasets/shunk031-private/CGL-Dataset-private/resolve/main/layout_imgs_6w_2.zip",
    ],
}

# The correspondence of the following category names
# is referred to https://tianchi.aliyun.com/dataset/142692#json-file-structure
CATEGORIES: Dict[str, str] = {
    "Logo": "logo",
    "文字": "text",
    "衬底": "underlay",
    "符号元素": "embellishment",
    "强调突出子部分文字": "highlighted text",
}


class UserSelectedValue(BaseModel):
    name: str


class Point(BaseModel):
    x: float
    y: float


class TextData(BaseModel):
    user_selected_value: UserSelectedValue = Field(alias="userSelectedValue")
    category_description: str = Field(alias="categoryDesc")
    points: List[Point]


class TextAnnotationTrainData(BaseModel):
    is_sample: bool = Field(alias="isSample")
    image: str
    rotate: float
    data: List[TextData]
    pin: str

    @property
    def id_images(self) -> int:
        # 'ali_anno_1/22.png' -> ['ali_anno_1', '22.png']
        _, id_filename = self.image.split("/")
        # 22.png -> ['22', '.png']
        root, _ = os.path.splitext(id_filename)
        return int(root)


class TextAnnotationTestData(BaseModel):
    image_filename: str
    product_detail_highlighted_word: Optional[List[str]] = Field(
        default=None, alias="productDetailHighlightWord"
    )
    blc_text: Optional[List[str]] = None
    adv_sellpoint: Optional[List[str]] = None


class TextFeatureData(BaseModel):
    feats: List[List[List[float]]]
    num: Optional[int] = None
    pos: Optional[List[Tuple[float, float, float, float]]] = None

    @model_validator(mode="after")
    def check_num_field(self) -> Self:
        if self.num is None:
            self.num = len(self.feats)

        assert self.num == len(self.feats)
        return self

    @model_validator(mode="after")
    def check_pos_field(self) -> Self:
        if self.pos:
            assert self.num == len(self.pos) == len(self.feats)
        return self


class CGLv2Processor(InstancesProcessor):
    def get_features_base_dict(self, is_ralf_style: bool):
        base_features = {
            "image_id": ds.Value("int64"),
            "file_name": ds.Value("string"),
            "width": ds.Value("int64"),
            "height": ds.Value("int64"),
        }
        image_features = (
            {
                "original_poster": ds.Image(),
                "inpainted_poster": ds.Image(),
            }
            if is_ralf_style
            else {
                "image": ds.Image(),
            }
        )
        return {**base_features, **image_features}

    def get_features_instance_dict(self, decode_rle: bool, rename_category_names: bool):
        category_names = (
            list(CATEGORIES.values())
            if rename_category_names
            else list(CATEGORIES.keys())
        )
        segmentation_feature = (
            ds.Image()
            if decode_rle
            else {
                "counts": ds.Value("binary"),
                "size": ds.Sequence(ds.Value("int32")),
            }
        )
        return {
            "annotation_id": ds.Value("int64"),
            "area": ds.Value("int64"),
            "bbox": ds.Sequence(ds.Value("int64")),
            "category": {
                "category_id": ds.Value("int64"),
                "name": ds.ClassLabel(
                    num_classes=len(category_names), names=category_names
                ),
                "supercategory": ds.Value("string"),
            },
            "category_id": ds.Value("int64"),
            "image_id": ds.Value("int64"),
            "iscrowd": ds.Value("bool"),
            "segmentation": segmentation_feature,
        }

    def get_features_text_annotations_dict(self):
        return {
            "is_sample": ds.Value("bool"),
            "image": ds.Value("string"),
            "rotate": ds.Value("float32"),
            "pin": ds.Value("string"),
            "data": ds.Sequence(
                {
                    "category_description": ds.Value("string"),
                    "points": ds.Sequence(
                        {
                            "x": ds.Value("int64"),
                            "y": ds.Value("int64"),
                        }
                    ),
                    "user_selected_value": {
                        "name": ds.Value("string"),
                    },
                }
            ),
            "product_detail_highlighted_word": ds.Sequence(ds.Value("string")),
            "blc_text": ds.Sequence(ds.Value("string")),
            "adv_sellpoint": ds.Sequence(ds.Value("string")),
        }

    def get_features_text_features_dict(self):
        return {
            "num": ds.Value("int64"),
            "pos": ds.Sequence(ds.Sequence(ds.Value("int64"))),
            "feats": ds.Sequence(ds.Sequence(ds.Sequence(ds.Value("float32")))),
        }

    def get_features(
        self, decode_rle: bool, rename_category_names: bool, is_ralf_style: bool
    ) -> ds.Features:
        features_dict = self.get_features_base_dict(is_ralf_style)
        annotations = ds.Sequence(
            self.get_features_instance_dict(
                decode_rle=decode_rle, rename_category_names=rename_category_names
            )
        )
        text_annotations = self.get_features_text_annotations_dict()
        text_features = self.get_features_text_features_dict()
        features_dict.update(
            {
                "annotations": annotations,
                "text_annotations": text_annotations,
                "text_features": text_features,
            }
        )
        return ds.Features(features_dict)

    def _load_train_texts_data(
        self,
        txt_path: pathlib.Path,
        image_dicts: List[JsonDict],
        tqdm_desc: str = "Load text annotations for training",
    ) -> Dict[ImageId, TextAnnotationTrainData]:
        assert txt_path.stem == "train", txt_path

        texts: Dict[ImageId, TextAnnotationTrainData] = {}
        with txt_path.open("r") as rf:
            for line in tqdm(rf, desc=tqdm_desc):
                text_dict = ast.literal_eval(line)
                text_data_ann = TextAnnotationTrainData(**text_dict)
                image_dict = image_dicts[text_data_ann.id_images]
                image_id = image_dict["id"]

                if image_id in texts:
                    raise ValueError(f"Duplicate image id: {image_id}")

                texts[image_id] = text_data_ann
        return texts

    def _load_test_texts_data(
        self,
        txt_path: pathlib.Path,
        images: Dict[ImageId, ImageData],
        tqdm_desc: str = "Load text annotations for test",
    ) -> Dict[ImageId, TextAnnotationTestData]:
        assert txt_path.stem == "test", txt_path
        images_dict = {image.file_name: image for image in images.values()}

        texts = {}
        with txt_path.open("r", encoding="utf-8") as rf:
            for line in tqdm(rf, desc=tqdm_desc):
                image_filename, json_str = line.split("\t")
                text_dict = ast.literal_eval(json_str)
                text_data_ann = TextAnnotationTestData(
                    image_filename=image_filename, **text_dict
                )
                image_id = images_dict[image_filename].image_id
                if image_id in texts:
                    raise ValueError(f"Duplicate image id: {image_id}")

                texts[image_id] = text_data_ann
        return texts

    def load_categories_data(
        self,
        category_dicts: List[JsonDict],
        rename_category_names: bool,
        category_data_class: Type[CategoryData] = CategoryData,
        tqdm_desc: str = "Load categories",
    ) -> Dict[CategoryId, CategoryData]:
        categories = {}
        for cat_dict in tqdm(category_dicts, desc=tqdm_desc):
            if rename_category_names:
                cat_dict["name"] = CATEGORIES[cat_dict["name"]]
                cat_dict["supercategory"] = CATEGORIES[cat_dict["supercategory"]]

            category_data = category_data_class(**cat_dict)
            categories[category_data.category_id] = category_data
        return categories

    def load_texts_data(
        self,
        txt_path: pathlib.Path,
        image_dicts: List[JsonDict],
        images: Dict[ImageId, ImageData],
    ) -> Union[
        Dict[ImageId, TextAnnotationTrainData], Dict[ImageId, TextAnnotationTestData]
    ]:
        if txt_path.stem == "train":
            return self._load_train_texts_data(
                txt_path=txt_path,
                image_dicts=image_dicts,
            )
        elif txt_path.stem == "test":
            return self._load_test_texts_data(
                txt_path=txt_path,
                images=images,
            )
        else:
            raise ValueError(f"Unknown text file: {txt_path}")

    def load_text_features(
        self,
        txt_feature_dir: pathlib.Path,
        image_dicts: List[JsonDict],
        tqdm_desc="Load text features",
    ) -> Dict[ImageId, TextFeatureData]:
        text_features = {}
        for image_dict in tqdm(image_dicts, desc=tqdm_desc):
            image_filename = image_dict["file_name"]
            root, _ = os.path.splitext(image_filename)
            txt_feature_path = txt_feature_dir / f"{root}_feats.pth"

            if not txt_feature_path.exists():
                # logger.warning(f"Text feature file not found: {txt_feature_path}")
                continue

            txt_feature_dict = torch.load(
                txt_feature_path, map_location=torch.device("cpu")
            )
            txt_feature_dict["feats"] = [
                f.numpy().tolist() for f in txt_feature_dict["feats"]
            ]
            txt_feature_data = TextFeatureData(**txt_feature_dict)

            image_id = image_dict["id"]
            if image_id in text_features:
                raise ValueError(f"Duplicate image id: {image_id}")

            text_features[image_id] = txt_feature_data
        return text_features

    def generate_examples(  # type: ignore[override]
        self,
        image_dir: PathLike,
        images: Dict[ImageId, ImageData],
        annotations: Dict[ImageId, List[InstancesAnnotationData]],
        categories: Dict[CategoryId, CategoryData],
        texts: Union[
            Dict[ImageId, TextAnnotationTrainData],
            Dict[ImageId, TextAnnotationTestData],
        ],
        text_features: Optional[Dict[ImageId, TextFeatureData]] = None,
        v1_image_files: Optional[Dict[str, pathlib.Path]] = None,
    ) -> Iterator[Tuple[int, InstanceExample]]:
        for idx, image_id in enumerate(images.keys()):
            image_data = images[image_id]
            image_anns = annotations[image_id]

            if len(image_anns) < 1:
                logger.warning(f"No annotation found for image id: {image_id}.")
                continue

            example = image_data.model_dump()

            if v1_image_files is not None:
                v1_file_name = restore_v1_filename(image_data.file_name)
                v1_file_path = v1_image_files.get(v1_file_name)

                if v1_file_path is not None:
                    original_poster = self.load_image(
                        image_path=v1_image_files[v1_file_name]
                    )
                    example["original_poster"] = original_poster

            inpainted_image = self.load_image(
                image_path=os.path.join(image_dir, image_data.file_name),
            )
            example["inpainted_poster"] = inpainted_image

            text_data = texts.get(image_id)
            example["text_annotations"] = text_data.model_dump() if text_data else None

            if text_features is not None:
                text_feature = text_features.get(image_id)
                example["text_features"] = (
                    text_feature.model_dump() if text_feature else None
                )

            example["annotations"] = []
            for ann in image_anns:
                ann_dict = ann.model_dump()
                category = categories[ann.category_id]
                ann_dict["category"] = category.model_dump()
                example["annotations"].append(ann_dict)

            yield idx, example  # type: ignore


def get_v1_image_file_paths(
    image_dirs: List[pathlib.Path],
) -> Dict[str, pathlib.Path]:
    image_file_paths: Dict[str, pathlib.Path] = {}

    for image_dir in image_dirs:
        jpg_files = list(image_dir.glob("**/*.jpg"))
        png_files = list(image_dir.glob("**/*.png"))
        image_files = jpg_files + png_files
        for image_file in image_files:
            assert image_file.name not in image_file_paths, image_file
            image_file_paths[image_file.name] = image_file

    return image_file_paths


def restore_v1_filename(file_name):
    # Restore the file name of the CGL dataset from the file name of the CGL v2 dataset
    # e.g.: 'O1CN01HnK3zH1HoH7oxbsE5_!!3409010804-0-alimamazszw_mask002.jpg'
    # -> 'O1CN01HnK3zH1HoH7oxbsE5_!!3409010804-0-alimamazszw.jpg'

    # ('O1CN01HnK3zH1HoH7oxbsE5_!!3409010804-0-alimamazszw_mask002', '.jpg')
    root, ext = os.path.splitext(file_name)

    # ['O1CN01HnK3zH1HoH7oxbsE5', '!!3409010804-0-alimamazszw', 'mask002']
    roots = root.split("_")

    # O1CN01HnK3zH1HoH7oxbsE5_!!3409010804-0-alimamazszw
    root = "_".join(roots[:2])

    # O1CN01HnK3zH1HoH7oxbsE5_!!3409010804-0-alimamazszw.jpg
    return f"{root}{ext}"


def ralf_style_example(
    example,
    inpainter: SimpleLama,
    saliency_testers: List[SaliencyTester],
    saliency_map_cols: Sequence[str],
):
    from ralfpt.inpainting import apply_inpainting
    from ralfpt.saliency_detection import apply_saliency_detection
    from ralfpt.transforms import has_valid_area, load_from_cgl_ltwh
    from ralfpt.typehints import Element

    def get_cgl_layout_elements(
        annotations, image_w: int, image_h: int
    ) -> List[Element]:
        elements = []

        for ann in annotations:
            category = ann["category"]
            label = category["name"]

            coordinates = load_from_cgl_ltwh(
                ltwh=ann["bbox"], global_width=image_w, global_height=image_h
            )
            if has_valid_area(**coordinates):
                element: Element = {"label": label, "coordinates": coordinates}
                elements.append(element)

        return elements

    assert len(saliency_testers) == len(saliency_map_cols)

    original_poster = example.get("original_poster")
    if original_poster is None:
        canvas = example["inpainted_poster"]
        saliency_maps = apply_saliency_detection(
            image=canvas,
            saliency_testers=saliency_testers,  # type: ignore
        )
        for sal_col, sal_map in zip(saliency_map_cols, saliency_maps):
            example[sal_col] = sal_map

        return example

    annotations = example["annotations"]
    is_test = len(annotations) < 1

    if is_test:
        return example

    image_w, image_h = example["width"], example["height"]

    elements = get_cgl_layout_elements(
        annotations=example["annotations"], image_w=image_w, image_h=image_h
    )

    #
    # Apply RALF-style inpainting
    #
    inpainted_image = apply_inpainting(
        image=original_poster, elements=elements, inpainter=inpainter
    )
    example["inpainted_poster"] = inpainted_image

    #
    # Apply Ralf-style saliency detection
    #
    saliency_maps = apply_saliency_detection(
        image=inpainted_image,
        saliency_testers=saliency_testers,  # type: ignore
    )
    for sal_col, sal_map in zip(saliency_map_cols, saliency_maps):
        example[sal_col] = sal_map

    return example


@dataclass
class CGLDatasetV2Config(ds.BuilderConfig):
    decode_rle: bool = False
    include_text_features: bool = False
    rename_category_names: bool = False
    processor: CGLv2Processor = CGLv2Processor()

    saliency_maps: Sequence[str] = (
        "saliency_map",
        "saliency_map_sub",
    )
    saliency_testers: Sequence[str] = (
        "creative-graphic-design/ISNet-general-use",
        "creative-graphic-design/BASNet-SmartText",
    )

    def __post_init__(self):
        super().__post_init__()
        assert len(self.saliency_maps) == len(self.saliency_testers)


class CGLDatasetV2(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = CGLDatasetV2Config
    BUILDER_CONFIGS = [
        CGLDatasetV2Config(name="default", version=VERSION, description=_DESCRIPTION),
        CGLDatasetV2Config(name="ralf", version=VERSION, description=_DESCRIPTION),
    ]

    def _info(self) -> ds.DatasetInfo:
        config: CGLDatasetV2Config = self.config  # type: ignore
        processor = config.processor
        features = processor.get_features(
            decode_rle=config.decode_rle,
            rename_category_names=config.rename_category_names,
            is_ralf_style=config.name == "ralf",
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(
        self, dl_manager: ds.DownloadManager
    ) -> List[ds.SplitGenerator]:
        base_dir_path = dl_manager.download_and_extract(_URLS["v2"])
        assert isinstance(base_dir_path, str)

        dir_path = pathlib.Path(base_dir_path) / "RADM_dataset"

        ann_dir = dir_path / "annotations"
        img_dir = dir_path / "images"
        txt_dir = dir_path / "texts"
        txt_feature_dir = dir_path / "text_features"

        tng_ann_json_path = ann_dir / "train.json"
        tst_ann_json_path = ann_dir / "test.json"

        tng_img_dir = img_dir / "train"
        tst_img_dir = img_dir / "test"

        tng_img_json_path = tng_img_dir / "train.json"
        tst_img_json_path = tst_img_dir / "test.json"

        tng_txt_path = txt_dir / "train.txt"
        tst_txt_path = txt_dir / "test.txt"

        tng_txt_feature_dir = txt_feature_dir / "train"
        tst_txt_feature_dir = txt_feature_dir / "test"

        v1_image_files = (
            get_v1_image_file_paths(
                image_dirs=[
                    pathlib.Path(d)
                    for d in dl_manager.download_and_extract(_URLS["v1"])
                ]
            )
            if self.config.name == "ralf"
            else None
        )
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "ann_json_path": tng_ann_json_path,
                    "img_dir": tng_img_dir,
                    "img_json_path": tng_img_json_path,
                    "txt_path": tng_txt_path,
                    "txt_feature_dir": tng_txt_feature_dir,
                    "v1_image_files": v1_image_files,
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={
                    "ann_json_path": tst_ann_json_path,
                    "img_dir": tst_img_dir,
                    "img_json_path": tst_img_json_path,
                    "txt_path": tst_txt_path,
                    "txt_feature_dir": tst_txt_feature_dir,
                },
            ),
        ]

    def _generate_examples(
        self,
        ann_json_path: pathlib.Path,
        img_dir: pathlib.Path,
        txt_path: pathlib.Path,
        txt_feature_dir: pathlib.Path,
        v1_image_files: Optional[Dict[str, pathlib.Path]] = None,
        **kwargs,
    ):
        config: CGLDatasetV2Config = self.config  # type: ignore
        processor: CGLv2Processor = config.processor

        ann_json = processor.load_annotation_json(
            ann_file_path=ann_json_path,
        )
        images = processor.load_images_data(
            image_dicts=ann_json["images"],
        )
        categories = processor.load_categories_data(
            category_dicts=ann_json["categories"],
            rename_category_names=config.rename_category_names,
        )
        texts = processor.load_texts_data(
            txt_path=txt_path, image_dicts=ann_json["images"], images=images
        )
        text_features = (
            processor.load_text_features(
                txt_feature_dir=txt_feature_dir,
                image_dicts=ann_json["images"],
            )
            if config.include_text_features  # type: ignore
            else None
        )
        annotations = processor.load_data(
            ann_dicts=ann_json["annotations"],
            images=images,
            decode_rle=config.decode_rle,
        )

        generator = processor.generate_examples(
            image_dir=img_dir,
            images=images,
            annotations=annotations,
            categories=categories,
            texts=texts,
            text_features=text_features,
            v1_image_files=v1_image_files,
        )

        def _generate_default(generator):
            for idx, example in generator:
                yield idx, example

        def _generate_ralf_style(generator):
            from ralfpt.saliency_detection import SaliencyTester
            from simple_lama_inpainting import SimpleLama

            inpainter = SimpleLama()
            saliency_testers = [
                SaliencyTester(model_name=model) for model in config.saliency_testers
            ]
            for idx, example in generator:
                example = ralf_style_example(
                    example,
                    inpainter=inpainter,
                    saliency_map_cols=config.saliency_maps,
                    saliency_testers=saliency_testers,
                )
                yield idx, example

        if config.name == "default":
            yield from _generate_default(generator)

        elif config.name == "ralf":
            yield from _generate_ralf_style(generator)

        else:
            raise ValueError(f"Invalid config name: {config.name}")
