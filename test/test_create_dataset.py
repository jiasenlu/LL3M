import sys
import os
sys.path.append(os.path.abspath(""))
from create_data.llava.llava_instruct_dataset_builder import llava_insturct_mix665k
import tensorflow_datasets as tfds

tfds.load('llava_insturct_mix665k', try_gcs=False, shuffle_files=False)
