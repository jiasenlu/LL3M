from typing import Iterator, Tuple, Any

import cv2
import glob
import h5py
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from module.conversion_utils import MultiThreadedDatasetBuilder
from PIL import Image
import random

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
            
IMAGE_PATH = 'datasets/LLaVA-Instruct/'
def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    
    # load the json file
    for sample in paths:
        
        if 'image' in sample:
            im = Image.open(os.path.join(IMAGE_PATH, sample['image']))
            im = im.convert('RGB')
            im =  np.array(im)
            has_image = 1
        else:
            im = np.zeros((336, 336, 3), dtype=np.uint8)
            has_image = 0

        new_conv = []
        conversation = sample.get('conversations', [])
        for conv in conversation:
            if conv['from'] == 'human':
                new_conv.append({'from': 'human', 'value': conv['value']})

            if conv['from'] == 'gpt':
                if 'value' in conv:
                    new_conv.append({'from': 'gpt', 'value': conv['value']})
                else:
                    new_conv.append({'from': 'gpt', 'value': conv['text']})

        yield random.getrandbits(256), {
            'image': im,
            'conversations': new_conv,
            'has_image': has_image,
        }
    
class llava_insturct_mix665k(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 20              # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 1000  # number of paths converted & stored in memory before writing to disk
                                # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                                # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'has_image': tfds.features.Scalar(dtype=tf.int32),
                'image': tfds.features.Image(shape=(None, None, 3)),
                'conversations': tfds.features.Sequence({
                    'from': tfds.features.Text(),
                    'value': tfds.features.Text(),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        print(self.info)
        return {
            'train': json.load(open('datasets/LLaVA-Instruct/llava_v1_5_mix665k.json', 'r')),
        }