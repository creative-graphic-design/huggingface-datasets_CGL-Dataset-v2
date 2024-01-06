import ast
import json
import os
import pathlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, TypedDict, Union

import datasets as ds
import numpy as np
from datasets.utils.logging import get_logger
from PIL import Image
from PIL.Image import Image as PilImage
from pycocotools import mask as cocomask
from tqdm import tqdm

logger = get_logger(__name__)

JsonDict = Dict[str, Any]
ImageId = int
CategoryId = int
AnnotationId = int
Bbox = Tuple[float, float, float, float]


_DESCRIPTION = """\
"""

_CITATION = """
@inproceedings{li2023relation,
  title={Relation-Aware Diffusion Model for Controllable Poster Layout Generation},
  author={Li, Fengheng and Liu, An and Feng, Wei and Zhu, Honghe and Li, Yaoyu and Zhang, Zheng and Lv, Jingjing and Zhu, Xin and Shen, Junjie and Lin, Zhangang},
  booktitle={Proceedings of the 32nd ACM international conference on information \& knowledge management},
  pages={1249--1258},
  year={2023}
}
"""

_HOMEPAGE = "https://github.com/liuan0803/RADM"

_LICENSE = """\
"""


class UncompressedRLE(TypedDict):
    counts: List[int]
    size: Tuple[int, int]


class CompressedRLE(TypedDict):
    counts: bytes
    size: Tuple[int, int]


@dataclass
class ImageData(object):
    image_id: ImageId
    file_name: str
    width: int
    height: int

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "ImageData":
        return cls(
            image_id=json_dict["id"],
            file_name=json_dict["file_name"],
            width=json_dict["width"],
            height=json_dict["height"],
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


@dataclass
class CategoryData(object):
    category_id: int
    name: str
    supercategory: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "CategoryData":
        return cls(
            category_id=json_dict["id"],
            name=json_dict["name"],
            supercategory=json_dict["supercategory"],
        )


@dataclass
class VisualAnnotationData(object):
    annotation_id: AnnotationId
    image_id: ImageId
    segmentation: Union[np.ndarray, CompressedRLE]
    area: float
    iscrowd: bool
    bbox: Bbox
    category_id: int

    @classmethod
    def compress_rle(
        cls,
        segmentation: Union[List[List[float]], UncompressedRLE],
        iscrowd: bool,
        height: int,
        width: int,
    ) -> CompressedRLE:
        if iscrowd:
            rle = cocomask.frPyObjects(segmentation, h=height, w=width)
        else:
            rles = cocomask.frPyObjects(segmentation, h=height, w=width)
            rle = cocomask.merge(rles)  # type: ignore

        return rle  # type: ignore

    @classmethod
    def rle_segmentation_to_binary_mask(
        cls, segmentation, iscrowd: bool, height: int, width: int
    ) -> np.ndarray:
        rle = cls.compress_rle(
            segmentation=segmentation, iscrowd=iscrowd, height=height, width=width
        )
        return cocomask.decode(rle)  # type: ignore

    @classmethod
    def rle_segmentation_to_mask(
        cls,
        segmentation: Union[List[List[float]], UncompressedRLE],
        iscrowd: bool,
        height: int,
        width: int,
    ) -> np.ndarray:
        binary_mask = cls.rle_segmentation_to_binary_mask(
            segmentation=segmentation, iscrowd=iscrowd, height=height, width=width
        )
        return binary_mask * 255

    @classmethod
    def from_dict(
        cls,
        json_dict: JsonDict,
        images: Dict[ImageId, ImageData],
        decode_rle: bool,
    ) -> "VisualAnnotationData":
        segmentation = json_dict["segmentation"]
        image_id = json_dict["image_id"]
        image_data = images[image_id]
        iscrowd = bool(json_dict["iscrowd"])

        segmentation_mask = (
            cls.rle_segmentation_to_mask(
                segmentation=segmentation,
                iscrowd=iscrowd,
                height=image_data.height,
                width=image_data.width,
            )
            if decode_rle
            else cls.compress_rle(
                segmentation=segmentation,
                iscrowd=iscrowd,
                height=image_data.height,
                width=image_data.width,
            )
        )
        return cls(
            annotation_id=json_dict["id"],
            image_id=image_id,
            segmentation=segmentation_mask,  # type: ignore
            area=json_dict["area"],
            iscrowd=iscrowd,
            bbox=json_dict["bbox"],
            category_id=json_dict["category_id"],
        )


@dataclass
class UserSelectedValue(object):
    name: str


@dataclass
class Point(object):
    x: int
    y: int


@dataclass
class TextData(object):
    user_selected_value: UserSelectedValue
    category_description: str
    points: List[Point]

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "TextData":
        return cls(
            user_selected_value=UserSelectedValue(**json_dict["userSelectedValue"]),
            points=[Point(**p) for p in json_dict["points"]],
            category_description=json_dict["categoryDesc"],
        )


@dataclass
class TextAnnotationData(object):
    is_sample: bool
    image: str
    rotate: float
    data: List[TextData]
    pin: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "TextAnnotationData":
        text_data = [TextData.from_dict(d) for d in json_dict["data"]]
        return cls(
            is_sample=bool(int(json_dict["isSample"])),
            image=json_dict["image"],
            rotate=json_dict["rotate"],
            pin=json_dict["pin"],
            data=text_data,
        )


def load_json(json_path: pathlib.Path) -> JsonDict:
    logger.info(f"Load from {json_path}")
    with json_path.open("r") as rf:
        json_dict = json.load(rf)
    return json_dict


def load_image(image_path: pathlib.Path) -> PilImage:
    logger.info(f"Load from {image_path}")
    return Image.open(image_path)


def load_images_data(
    image_dicts: List[JsonDict],
    tqdm_desc="Load images",
) -> Dict[ImageId, ImageData]:
    images = {}
    for image_dict in tqdm(image_dicts, desc=tqdm_desc):
        image_data = ImageData.from_dict(image_dict)
        images[image_data.image_id] = image_data
    return images


def load_categories_data(
    category_dicts: List[JsonDict],
    tqdm_desc: str = "Load categories",
) -> Dict[CategoryId, CategoryData]:
    categories = {}
    for category_dict in tqdm(category_dicts, desc=tqdm_desc):
        category_data = CategoryData.from_dict(category_dict)
        categories[category_data.category_id] = category_data
    return categories


def load_texts_data(txt_path: pathlib.Path) -> List[TextAnnotationData]:
    text_annotations = []
    with txt_path.open("r") as rf:
        for line in rf:
            text_dict = ast.literal_eval(line)
            text_data_ann = TextAnnotationData.from_dict(text_dict)
            text_annotations.append(text_data_ann)
    return text_annotations


def load_annotation_data(
    label_dicts: List[JsonDict],
    images: Dict[ImageId, ImageData],
    decode_rle: bool,
    tqdm_desc: str = "Load annotation data",
) -> Dict[ImageId, List[VisualAnnotationData]]:
    labels = defaultdict(list)
    label_dicts = sorted(label_dicts, key=lambda d: d["image_id"])

    for label_dict in tqdm(label_dicts, desc=tqdm_desc):
        label_data = VisualAnnotationData.from_dict(
            label_dict, images=images, decode_rle=decode_rle
        )
        labels[label_data.image_id].append(label_data)
    return labels


@dataclass
class CGLDatasetV2Config(ds.BuilderConfig):
    decode_rle: bool = False


class CGLDatasetV2(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = CGLDatasetV2Config
    BUILDER_CONFIGS = [
        CGLDatasetV2Config(version=VERSION, description=_DESCRIPTION),
    ]

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features()
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
        assert dl_manager.manual_dir is not None
        base_dir_path = os.path.expanduser(dl_manager.manual_dir)

        if not os.path.exists(base_dir_path):
            raise FileNotFoundError()

        base_dir_path = dl_manager.extract(base_dir_path)
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

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "ann_json_path": tng_ann_json_path,
                    "img_dir": tng_img_dir,
                    "img_json_path": tng_img_json_path,
                    "txt_path": tng_txt_path,
                    "txt_feature_dir": tng_txt_feature_dir,
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
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
        img_json_path: pathlib.Path,
        txt_path: pathlib.Path,
        txt_feature_dir: pathlib.Path,
    ):
        ann_json = load_json(ann_json_path)
        images = load_images_data(image_dicts=ann_json["images"])
        texts = load_texts_data(txt_path=txt_path)
        categories = load_categories_data(category_dicts=ann_json["categories"])

        annotations = load_annotation_data(
            label_dicts=ann_json["annotations"],
            images=images,
            decode_rle=self.config.decode_rle,
        )

        for idx, image_id in enumerate(images.keys()):
            image_data = images[image_id]
            image_anns = annotations[image_id]

            image = load_image(image_path=img_dir / image_data.file_name)
            example = asdict(image_data)
            example["image"] = image

            example["annotations"] = []
            for ann in image_anns:
                ann_dict = asdict(ann)
                category = categories[ann.category_id]
                ann_dict["category"] = asdict(category)
                example["annotations"].append(ann_dict)

            breakpoint()
