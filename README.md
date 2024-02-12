---
annotations_creators:
- crowdsourced
language:
- zh
language_creators:
- found
license:
- unknown
multilinguality:
- monolingual
pretty_name: CGL-Dataset v2
size_categories: []
source_datasets:
- CGL-Dataset
tags:
- graphic design
task_categories:
- other
task_ids: []
---

# Dataset Card for CGL-Dataset-v2

[![CI](https://github.com/shunk031/huggingface-datasets_CGL-Dataset-v2/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/huggingface-datasets_CGL-Dataset-v2/actions/workflows/ci.yaml)
[![Sync HF](https://github.com/shunk031/huggingface-datasets_CGL-Dataset-v2/actions/workflows/push_to_hub.yaml/badge.svg)](https://github.com/shunk031/huggingface-datasets_CGL-Dataset-v2/actions/workflows/push_to_hub.yaml)

## Table of Contents
- [Dataset Card Creation Guide](#dataset-card-creation-guide)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://github.com/liuan0803/RADM
- **Repository:** https://github.com/shunk031/huggingface-datasets_CGL-Dataset-v2
- **Paper (Preprint):** https://arxiv.org/abs/2306.09086
- **Paper (CIKM'23):** https://dl.acm.org/doi/10.1145/3583780.3615028

### Dataset Summary

CGL-Dataset V2 is a dataset for the task of automatic graphic layout design of advertising posters, containing 60,548 training samples and 1035 testing samples. It is an extension of CGL-Dataset.

### Supported Tasks and Leaderboards

[More Information Needed]

<!-- For each of the tasks tagged for this dataset, give a brief description of the tag, metrics, and suggested models (with a link to their HuggingFace implementation if available). Give a similar description of tasks that were not covered by the structured tag set (repace the `task-category-tag` with an appropriate `other:other-task-name`).

- `task-category-tag`: The dataset can be used to train a model for [TASK NAME], which consists in [TASK DESCRIPTION]. Success on this task is typically measured by achieving a *high/low* [metric name](https://huggingface.co/metrics/metric_name). The ([model name](https://huggingface.co/model_name) or [model class](https://huggingface.co/transformers/model_doc/model_class.html)) model currently achieves the following score. *[IF A LEADERBOARD IS AVAILABLE]:* This task has an active leaderboard which can be found at [leaderboard url]() and ranks models based on [metric name](https://huggingface.co/metrics/metric_name) while also reporting [other metric name](https://huggingface.co/metrics/other_metric_name). -->

### Languages

The language data in CGL-Dataset v2 is in Chinese ([BCP-47 zh](https://www.rfc-editor.org/info/bcp47)).

## Dataset Structure

### Data Instances

To use CGL-Dataset v2 dataset, you need to download `RADM_dataset.tar.gz` that includes the poster image, text and text features via [JD Cloud](https://3.cn/10-dQKDKG) or [Google Drive](https://drive.google.com/file/d/1ezOzR7MX3MFFIfWgJmmEaqXn3iDFp2si/view?usp=sharing).
Then place the downloaded files in the following structure and specify its path.

```shell
/path/to/datasets
└── RADM_dataset.tar.gz
```

```python
import datasets as ds

dataset = ds.load_dataset(
    path="shunk031/PosterErase",
    data_dir="/path/to/datasets/RADM_dataset.tar.gz",
    decode_rle=True, # True if Run-length Encoding (RLE) is to be decoded and converted to binary mask.
    include_text_features=True, # True if RoBERTa-based text feature is to be loaded.
)
```

### Data Fields

[More Information Needed]

<!-- List and describe the fields present in the dataset. Mention their data type, and whether they are used as input or output in any of the tasks the dataset currently supports. If the data has span indices, describe their attributes, such as whether they are at the character level or word level, whether they are contiguous or not, etc. If the datasets contains example IDs, state whether they have an inherent meaning, such as a mapping to other datasets or pointing to relationships between data points.

- `example_field`: description of `example_field`

Note that the descriptions can be initialized with the **Show Markdown Data Fields** output of the [Datasets Tagging app](https://huggingface.co/spaces/huggingface/datasets-tagging), you will then only need to refine the generated descriptions. -->

### Data Splits

[More Information Needed]

<!-- Describe and name the splits in the dataset if there are more than one.

Describe any criteria for splitting the data, if used. If there are differences between the splits (e.g. if the training annotations are machine-generated and the dev and test ones are created by humans, or if different numbers of annotators contributed to each example), describe them here.

Provide the sizes of each split. As appropriate, provide any descriptive statistics for the features, such as average length.  For example:

|                         | train | validation | test |
|-------------------------|------:|-----------:|-----:|
| Input Sentences         |       |            |      |
| Average Sentence Length |       |            |      | -->

## Dataset Creation

### Curation Rationale

[More Information Needed]

<!-- What need motivated the creation of this dataset? What are some of the reasons underlying the major choices involved in putting it together? -->

### Source Data

[More Information Needed]

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences,...) -->

#### Initial Data Collection and Normalization

[More Information Needed]

<!-- Describe the data collection process. Describe any criteria for data selection or filtering. List any key words or search terms used. If possible, include runtime information for the collection process.

If data was collected from other pre-existing datasets, link to source here and to their [Hugging Face version](https://huggingface.co/datasets/dataset_name).

If the data was modified or normalized after being collected (e.g. if the data is word-tokenized), describe the process and the tools used. -->

#### Who are the source language producers?

[More Information Needed]

<!-- State whether the data was produced by humans or machine generated. Describe the people or systems who originally created the data.

If available, include self-reported demographic or identity information for the source data creators, but avoid inferring this information. Instead state that this information is unknown. See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender.

Describe the conditions under which the data was created (for example, if the producers were crowdworkers, state what platform was used, or if the data was found, what website the data was found on). If compensation was provided, include that information here.

Describe other people represented or mentioned in the data. Where possible, link to references for the information. -->

### Annotations

[More Information Needed]

<!-- If the dataset contains annotations which are not part of the initial data collection, describe them in the following paragraphs. -->

#### Annotation process

[More Information Needed]

<!-- If applicable, describe the annotation process and any tools used, or state otherwise. Describe the amount of data annotated, if not all. Describe or reference annotation guidelines provided to the annotators. If available, provide interannotator statistics. Describe any annotation validation processes. -->

#### Who are the annotators?

[More Information Needed]

<!-- If annotations were collected for the source data (such as class labels or syntactic parses), state whether the annotations were produced by humans or machine generated.

Describe the people or systems who originally created the annotations and their selection criteria if applicable.

If available, include self-reported demographic or identity information for the annotators, but avoid inferring this information. Instead state that this information is unknown. See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender.

Describe the conditions under which the data was annotated (for example, if the annotators were crowdworkers, state what platform was used, or if the data was found, what website the data was found on). If compensation was provided, include that information here. -->

### Personal and Sensitive Information

[More Information Needed]

<!-- State whether the dataset uses identity categories and, if so, how the information is used. Describe where this information comes from (i.e. self-reporting, collecting from profiles, inferring, etc.). See [Larson 2017](https://www.aclweb.org/anthology/W17-1601.pdf) for using identity categories as a variables, particularly gender. State whether the data is linked to individuals and whether those individuals can be identified in the dataset, either directly or indirectly (i.e., in combination with other data).

State whether the dataset contains other data that might be considered sensitive (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history).  

If efforts were made to anonymize the data, describe the anonymization process. -->

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

<!-- Please discuss some of the ways you believe the use of this dataset will impact society.

The statement should include both positive outlooks, such as outlining how technologies developed through its use may improve people's lives, and discuss the accompanying risks. These risks may range from making important decisions more opaque to people who are affected by the technology, to reinforcing existing harmful biases (whose specifics should be discussed in the next section), among other considerations.

Also describe in this section if the proposed dataset contains a low-resource or under-represented language. If this is the case or if this task has any impact on underserved communities, please elaborate here. -->

### Discussion of Biases

[More Information Needed]

<!-- Provide descriptions of specific biases that are likely to be reflected in the data, and state whether any steps were taken to reduce their impact.

For Wikipedia text, see for example [Dinan et al 2020 on biases in Wikipedia (esp. Table 1)](https://arxiv.org/abs/2005.00614), or [Blodgett et al 2020](https://www.aclweb.org/anthology/2020.acl-main.485/) for a more general discussion of the topic.

If analyses have been run quantifying these biases, please add brief summaries and links to the studies here. -->

### Other Known Limitations

[More Information Needed]

<!-- If studies of the datasets have outlined other limitations of the dataset, such as annotation artifacts, please outline and cite them here. -->

## Additional Information

### Dataset Curators

[More Information Needed]

<!-- List the people involved in collecting the dataset and their affiliation(s). If funding information is known, include it here. -->

### Licensing Information

[More Information Needed]

<!-- Provide the license and link to the license webpage if available. -->

### Citation Information

<!-- Provide the [BibTex](http://www.bibtex.org/)-formatted reference for the dataset. For example:
```
@article{article_id,
  author    = {Author List},
  title     = {Dataset Paper Title},
  journal   = {Publication Venue},
  year      = {2525}
}
```

If the dataset has a [DOI](https://www.doi.org/), please provide it here. -->

```bibtex
@inproceedings{li2023relation,
  title={Relation-Aware Diffusion Model for Controllable Poster Layout Generation},
  author={Li, Fengheng and Liu, An and Feng, Wei and Zhu, Honghe and Li, Yaoyu and Zhang, Zheng and Lv, Jingjing and Zhu, Xin and Shen, Junjie and Lin, Zhangang},
  booktitle={Proceedings of the 32nd ACM international conference on information & knowledge management},
  pages={1249--1258},
  year={2023}
}
```

### Contributions

Thanks to [@liuan0803](https://github.com/liuan0803) for creating this dataset.
