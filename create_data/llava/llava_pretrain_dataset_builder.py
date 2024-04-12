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

IMAGE_PATH = 'datasets/LLaVA-Pretrain/images'
def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    
    # load the json file
    for sample in paths:
        im = Image.open(os.path.join(IMAGE_PATH, sample['image']))
        im = im.convert('RGB')
        
        yield sample['id'], {
            'image': np.array(im),
            'conversations': sample.get('conversations', []),
        }
    
class llava_pretrain(MultiThreadedDatasetBuilder):
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
                'image': tfds.features.Image(shape=(None, None, 3)),
                'conversations': tfds.features.Sequence({
                    'from': tfds.features.Text(),
                    'value': tfds.features.Text(),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        print(self.info)
        
        import pdb; pdb.set_tracer()
        return {
            'train': json.load(open('datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json', 'r')),
        }