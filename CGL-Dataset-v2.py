import ast
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union

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
    def get_features_base_dict(self):
        return {
            "image_id": ds.Value("int64"),
            "file_name": ds.Value("string"),
            "width": ds.Value("int64"),
            "height": ds.Value("int64"),
            "image": ds.Image(),
        }

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
        self, decode_rle: bool, rename_category_names: bool
    ) -> ds.Features:
        features_dict = self.get_features_base_dict()
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
        with txt_path.open("r") as rf:
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
        category_data_class: Type[CategoryData] = CategoryData,
        tqdm_desc: str = "Load categories",
        rename_category_names: bool = False,
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
    ) -> Iterator[Tuple[int, InstanceExample]]:
        for idx, image_id in enumerate(images.keys()):
            image_data = images[image_id]
            image_anns = annotations[image_id]

            if len(image_anns) < 1:
                logger.warning(f"No annotation found for image id: {image_id}.")
                continue

            image = self.load_image(
                image_path=os.path.join(image_dir, image_data.file_name),
            )
            example = image_data.model_dump()
            example["image"] = image

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


@dataclass
class CGLDatasetV2Config(ds.BuilderConfig):
    decode_rle: bool = False
    include_text_features: bool = False
    rename_category_names: bool = False
    processor: CGLv2Processor = CGLv2Processor()


class CGLDatasetV2(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = CGLDatasetV2Config
    BUILDER_CONFIGS = [
        CGLDatasetV2Config(version=VERSION, description=_DESCRIPTION),
    ]

    def _info(self) -> ds.DatasetInfo:
        config: CGLDatasetV2Config = self.config  # type: ignore
        processor = config.processor
        features = processor.get_features(
            decode_rle=config.decode_rle,
            rename_category_names=config.rename_category_names,
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
        txt_path: pathlib.Path,
        txt_feature_dir: pathlib.Path,
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

        yield from processor.generate_examples(
            image_dir=img_dir,
            images=images,
            annotations=annotations,
            categories=categories,
            texts=texts,
            text_features=text_features,
        )
