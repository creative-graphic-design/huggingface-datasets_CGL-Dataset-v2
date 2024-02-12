import ast
import json
import os
import pathlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import datasets as ds
import numpy as np
import torch
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

    @property
    def id_images(self) -> int:
        # 'ali_anno_1/22.png' -> ['ali_anno_1', '22.png']
        _, id_filename = self.image.split("/")
        root, _ = os.path.splitext(id_filename)
        return int(root)

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


@dataclass
class TextAnnotationTestData(object):
    image_filename: str
    product_detail_highlighted_word: Optional[List[str]] = None
    blc_text: Optional[List[str]] = None
    adv_sellpoint: Optional[List[str]] = None

    @classmethod
    def from_dict(
        cls, image_filename: str, json_dict: JsonDict
    ) -> "TextAnnotationTestData":
        return cls(
            image_filename=image_filename,
            product_detail_highlighted_word=json_dict.get("productDetailHighlightWord"),
            blc_text=json_dict.get("blc_text"),
            adv_sellpoint=json_dict.get("adv_sellpoint"),
        )


@dataclass
class TextFeatureData(object):
    feats: List[torch.Tensor]
    num: Optional[int] = None
    pos: Optional[List[Tuple[int, int, int, int]]] = None

    def __post_init__(self):
        if self.num is None:
            self.num = len(self.feats)

        assert self.num == len(self.feats)

        if self.pos:
            assert self.num == len(self.pos) == len(self.feats)


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


def load_train_texts_data(
    txt_path: pathlib.Path,
    image_dicts: List[JsonDict],
    tqdm_desc: str = "Load text annotations for training",
) -> Dict[ImageId, TextAnnotationData]:
    assert txt_path.stem == "train", txt_path

    texts: Dict[ImageId, TextAnnotationData] = {}
    with txt_path.open("r") as rf:
        for line in tqdm(rf, desc=tqdm_desc):
            text_dict = ast.literal_eval(line)
            text_data_ann = TextAnnotationData.from_dict(text_dict)
            image_dict = image_dicts[text_data_ann.id_images]
            image_id = image_dict["id"]

            if image_id in texts:
                raise ValueError(f"Duplicate image id: {image_id}")

            texts[image_id] = text_data_ann
    return texts


def load_test_texts_data(
    txt_path: pathlib.Path,
    images: Dict[ImageId, ImageData],
    tqdm_desc: str = "Load text annotations for test",
) -> Dict[ImageId, TextAnnotationTestData]:
    assert txt_path.stem == "test", txt_path
    images_dict = {image.file_name: image for image in images.values()}

    texts = {}
    with txt_path.open("r") as rf:
        for line in tqdm(rf, desc=tqdm_desc):
            image_filename, json_str = line.split("\t")
            text_dict = ast.literal_eval(json_str)
            text_data_ann = TextAnnotationTestData.from_dict(image_filename, text_dict)

            image_id = images_dict[image_filename].image_id
            if image_id in texts:
                raise ValueError(f"Duplicate image id: {image_id}")

            texts[image_id] = text_data_ann
    return texts


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


def load_text_features(
    txt_feature_dir: pathlib.Path,
    image_dicts: List[JsonDict],
    tqdm_desc="Load text features",
) -> Dict[ImageId, TextFeatureData]:
    text_features = {}
    for image_dict in tqdm(image_dicts):
        image_filename = image_dict["file_name"]
        root, _ = os.path.splitext(image_filename)
        txt_feature_path = txt_feature_dir / f"{root}_feats.pth"

        if not txt_feature_path.exists():
            # logger.warning(f"Text feature file not found: {txt_feature_path}")
            continue

        txt_feature_dict = torch.load(
            txt_feature_path, map_location=torch.device("cpu")
        )
        txt_feature_data = TextFeatureData(**txt_feature_dict)

        image_id = image_dict["id"]
        if image_id in text_features:
            raise ValueError(f"Duplicate image id: {image_id}")

        text_features[image_id] = txt_feature_data
    return text_features


@dataclass
class CGLDatasetV2Config(ds.BuilderConfig):
    decode_rle: bool = False
    include_text_features: bool = False


class CGLDatasetV2(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = CGLDatasetV2Config
    BUILDER_CONFIGS = [
        CGLDatasetV2Config(version=VERSION, description=_DESCRIPTION),
    ]

    def _info(self) -> ds.DatasetInfo:
        segmentation_feature = (
            ds.Image()
            if self.config.decode_rle  # type: ignore
            else {
                "counts": ds.Value("binary"),
                "size": ds.Sequence(ds.Value("int32")),
            }
        )
        features = ds.Features(
            {
                "image_id": ds.Value("int64"),
                "file_name": ds.Value("string"),
                "width": ds.Value("int64"),
                "height": ds.Value("int64"),
                "image": ds.Image(),
                "annotations": ds.Sequence(
                    {
                        "annotation_id": ds.Value("int64"),
                        "area": ds.Value("int64"),
                        "bbox": ds.Sequence(ds.Value("int64")),
                        "category": {
                            "category_id": ds.Value("int64"),
                            "name": ds.Value("string"),
                            "supercategory": ds.Value("string"),
                        },
                        "category_id": ds.Value("int64"),
                        "image_id": ds.Value("int64"),
                        "iscrowd": ds.Value("bool"),
                        "segmentation": segmentation_feature,
                    }
                ),
                "text_annotation": {
                    "is_sample": ds.Value("bool"),
                    "image": ds.Value("string"),
                    "rotate": ds.Value("float32"),
                    "pin": ds.Value("string"),
                    "data": ds.Sequence(
                        {
                            "category_description": ds.Value("string"),
                            "points": ds.Sequence(
                                {"x": ds.Value("int64"), "y": ds.Value("int64")}
                            ),
                            "user_selected_value": {"name": ds.Value("string")},
                        }
                    ),
                    "product_detail_highlighted_word": ds.Sequence(ds.Value("string")),
                    "blc_text": ds.Sequence(ds.Value("string")),
                    "adv_sellpoint": ds.Sequence(ds.Value("string")),
                },
                "text_feature": {
                    "num": ds.Value("int64"),
                    "pos": ds.Sequence(ds.Sequence(ds.Value("int64"))),
                    "feats": ds.Sequence(ds.Sequence(ds.Sequence(ds.Value("float32")))),
                },
            }
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
        assert dl_manager.manual_dir is not None
        base_dir_path = os.path.expanduser(dl_manager.manual_dir)

        if not os.path.exists(base_dir_path):
            raise FileNotFoundError()

        base_dir_path = dl_manager.extract(base_dir_path)
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

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={
                    "ann_json_path": tng_ann_json_path,
                    "img_dir": tng_img_dir,
                    "img_json_path": tng_img_json_path,
                    "txt_path": tng_txt_path,
                    "txt_feature_dir": tng_txt_feature_dir,
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
        img_json_path: pathlib.Path,
        txt_path: pathlib.Path,
        txt_feature_dir: pathlib.Path,
    ):
        ann_json = load_json(ann_json_path)
        images = load_images_data(image_dicts=ann_json["images"])
        categories = load_categories_data(category_dicts=ann_json["categories"])

        if txt_path.stem == "train":
            texts = load_train_texts_data(
                txt_path=txt_path, image_dicts=ann_json["images"]
            )
        elif txt_path.stem == "test":
            texts = load_test_texts_data(txt_path=txt_path, images=images)
        else:
            raise ValueError(f"Unknown text file: {txt_path}")

        text_features = (
            load_text_features(
                txt_feature_dir=txt_feature_dir, image_dicts=ann_json["images"]
            )
            if self.config.include_text_features  # type: ignore
            else None
        )

        annotations = load_annotation_data(
            label_dicts=ann_json["annotations"],
            images=images,
            decode_rle=self.config.decode_rle,  # type: ignore
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

            text_data = texts.get(image_id)
            example["text_annotation"] = asdict(text_data) if text_data else None

            if text_features:
                text_feature = text_features.get(image_id)
                example["text_feature"] = asdict(text_feature) if text_feature else None

            yield idx, example
