# Model that can be imported to register all tasks
import functools
import os.path

import seqio
import tensorflow as tf
from seqio import TaskRegistry
import os

from seqio import dataset_providers

from data.data_utils import get_default_vocabulary
from .preprocessors import *

TFDS_DATA_DIR = "gs://unified-io-2-us-east/"
MULTITASK_TFDS_DATA_DIR = f"{TFDS_DATA_DIR}multitask-datasets"

MULTIMODAL_OUTPUT_FEATURES = {
  "targets": dataset_providers.Feature(get_default_vocabulary()),
  "decoder_loss_weights": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
  "images": seqio.ContinuousFeature(dtype=tf.float32, rank=3),
  "image_input_idx": seqio.ContinuousFeature(dtype=tf.int32, rank=2),
}

TEXT_OUTPUT_FEATURES = {
  "targets": dataset_providers.Feature(get_default_vocabulary()),
  # "decoder_loss_weights": seqio.ContinuousFeature(dtype=tf.int32, rank=1),
}

def add_flan(name):
  full_name = f"flan2_{name.lower()}"
  TaskRegistry.add(
    full_name,
    source=seqio.TfdsDataSource(
      tfds_name=f"{full_name}:1.0.0",
      tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
      splits={
        "train": "train[2000:]",
        "validation": "train[:2000]"
      }
    ),
    preprocessors=[
      functools.partial(flan_preprocessor, name=full_name),
    ],
    output_features=TEXT_OUTPUT_FEATURES,
  )


FLAN_DATASETS = ["Flan2021", "T0", "NIv2", "CoT", "Dialog"]

for dataset in FLAN_DATASETS:
  add_flan(dataset)


TFRECORD_TEXT_FEATURES = {
  'id': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
  'text':tf.io.FixedLenFeature(shape=(), dtype=tf.string),
}

TaskRegistry.add(
  "c4",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join(TFDS_DATA_DIR, "dolma", "c4", "1.0.0", "c4-train*"),
    },
    feature_description=TFRECORD_TEXT_FEATURES,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "targets": ["text"],
      }),
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    full_lm
  ],
  output_features=TEXT_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "cc_en",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join(TFDS_DATA_DIR, "dolma", "cc*", "1.0.0", "cc_en_*"),
    },
    feature_description=TFRECORD_TEXT_FEATURES,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "targets": ["text"],
      }),
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    full_lm
  ],
  output_features=TEXT_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "peS2o",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join(TFDS_DATA_DIR, "dolma", "pes2o", "1.0.0", "pes2o-train*"),
    },
    feature_description=TFRECORD_TEXT_FEATURES,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "targets": ["text"],
      }),
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    full_lm
  ],
  output_features=TEXT_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "stack",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join(TFDS_DATA_DIR, "dolma", "stack", "1.0.0", "stack-train*"),
    },
    feature_description=TFRECORD_TEXT_FEATURES,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "targets": ["text"],
      }),
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    full_lm
  ],
  output_features=TEXT_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "reddit",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join(TFDS_DATA_DIR, "dolma", "reddit", "1.0.0", "reddit-train*"),
    },
    feature_description=TFRECORD_TEXT_FEATURES,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "targets": ["text"],
      }),
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    full_lm
  ],
  output_features=TEXT_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "wiki",
  source=seqio.TFExampleDataSource(
    split_to_filepattern={
      "train": os.path.join(TFDS_DATA_DIR, "dolma", "wiki", "1.0.0", "wiki-train*"),
    },
    feature_description=TFRECORD_TEXT_FEATURES,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "targets": ["text"],
      }),
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    full_lm
  ],
  output_features=TEXT_OUTPUT_FEATURES,
)

TaskRegistry.add(
  "coco_caption_2017",
  source=seqio.TfdsDataSource(
    tfds_name="coco_all:1.0.1",
    tfds_data_dir=MULTITASK_TFDS_DATA_DIR,
  ),
  preprocessors=[
    functools.partial(
      rekey, key_map={
        "image/filename": ["image/filename"], 
        "image": ["image"],
        "text": ["captions", "text"]
      }),
    functools.partial(
      multimodal_preprocessor,
      prompt_type="mistral",
      flatten_by_caption=True,
    ),
  ],
  output_features=MULTIMODAL_OUTPUT_FEATURES,
)