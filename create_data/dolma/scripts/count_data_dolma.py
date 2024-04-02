# count the number of lines in dolma. 
import sys

import argparse
import hashlib
import io
import json
import os
import random
import numpy as np
from tempfile import TemporaryDirectory
from google.cloud import storage
import tensorflow as tf
import gzip
import glob

def create_base_parser():
    parser = argparse.ArgumentParser(description='SCRAPE!')
    parser.add_argument(
        '-fold',
        dest='fold',
        default=0,
        type=int,
        help='which fold we are on'
    )
    parser.add_argument(
        '-seed',
        dest='seed',
        default=1337,
        type=int,
        help='which seed to use'
    )
    parser.add_argument(
        '-split',
        dest='split',
        default='train',
        type=str,
        help='which split to use'
    )
    parser.add_argument(
        '-base_fn',
        dest='base_fn',
        type=str,
        help='Base filename to use. You can start this with gs:// and we\'ll put it on google cloud.'
    )
    return parser

parser = create_base_parser()
parser.add_argument(
    '-data_dir',
    dest='data_dir',
    type=str,
    help='directory.'
)

args = parser.parse_args()

if 'stack-code' in args.data_dir:
    files = glob.glob(args.data_dir + '/**/' + '*.json.gz')
else:
    files = tf.io.gfile.listdir(args.data_dir)

files.sort()
idx_to_file = {i:cls_name for i, cls_name in enumerate(files)}


if 'stack-code' in args.data_dir:
    file_path = idx_to_file[args.fold]
else:
    directory = os.path.expanduser(args.data_dir)
    file_path = os.path.join(directory, idx_to_file[args.fold])
    
print("reading {}".format(file_path), flush=True)

count = 0
with tf.io.gfile.GFile(file_path, 'rb') as file:
    with gzip.GzipFile(fileobj=file) as gzip_file:
        for line in gzip_file:
            count += 1

# write to count json file.
out_file = file_path.replace('data', 'count')
out_file = out_file.replace('.gz', '')
out_path = '/'.join(out_file.split('/')[:-1])
os.makedirs(out_path, exist_ok=True)
json.dump({'count': count}, open(out_file, 'w'))
