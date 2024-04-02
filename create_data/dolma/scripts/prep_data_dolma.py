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
import math
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
        '-num_folds',
        dest='num_folds',
        default=1,
        type=int,
        help='Number of folds (corresponding to both the number of training files and the number of testing files)',
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

parser.add_argument(
    '-folder_name',
    type=str,
    help='folder name of the file.'
)

parser.add_argument(
    '-file_prefix',
    type=str,
    help='prefix of the file.'
)

args = parser.parse_args()
random.seed(args.seed)
out_fn = os.path.join(args.base_fn, args.folder_name, '1.0.0', '{}-{}.tfrecord-{:05d}-of-{:05d}'.format(args.file_prefix, args.split, args.fold, args.num_folds))

# if this is not exist.
file_exist = False
if out_fn.startswith('gs://'):
  bucket_name, file_name = out_fn.split('gs://', 1)[1].split('/', 1)
  gclient = storage.Client()
  bucket = gclient.bucket(bucket_name)
  file_exist = storage.Blob(bucket=bucket, name=file_name).exists(gclient)

class GCSTFRecordWriter(object):
    def __init__(self, fn, auto_close=False, options=None):
        """
        Shuffle things in the shuffle buffer and write to tfrecords
        If buffer_size == 0 then no shuffling
        :param fn:
        :param buffer_size:
        """
        self.fn = fn
        if fn.startswith('gs://'):
            self.gclient = storage.Client()
            self.storage_dir = TemporaryDirectory()
            self.writer = tf.io.TFRecordWriter(os.path.join(self.storage_dir.name, 'temp.tfrecord'), options=options)
            self.bucket_name, self.file_name = self.fn.split('gs://', 1)[1].split('/', 1)

        else:
            self.gclient = None
            self.bucket_name = None
            self.file_name = None
            self.storage_dir = None
            self.writer = tf.io.TFRecordWriter(fn, options=options)
        self.auto_close=auto_close

    def write(self, x):
        self.writer.write(x)

    def close(self):
        self.writer.close()
        if self.gclient is not None:
            print("UPLOADING!!!!!", flush=True)
            bucket = self.gclient.get_bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.upload_from_filename(os.path.join(self.storage_dir.name, 'temp.tfrecord'))
            self.storage_dir.cleanup()

    def __enter__(self):
        # Called when entering "with" context.
        return self

    def __exit__(self, *_):
        # Called when exiting "with" context.
        # Upload shit
        if self.auto_close:
            print("CALLING CLOSE")
            self.close()

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _make_dataset(args, directory, class_to_idx, is_valid_file=None):
  instances = []
  directory = os.path.expanduser(directory)
  
  file_count = {}
  for target_class in sorted(class_to_idx.keys()):
    class_index = class_to_idx[target_class]
    if 'stack-code' in args.data_dir:
        file_path = target_class
    else:
      file_path = os.path.join(directory, target_class)

    count_file = file_path.replace('data', 'count')
    count_file = count_file.replace('.gz', '')
    file_count[target_class] = json.load(open(count_file, 'r'))['count']
  
  # divided the file count into each shards.
  total_count = sum(file_count.values())
  shards_count = math.ceil(total_count / args.num_folds)
  
  c_name = []
  s_cnt = []
  e_cnt = []
  
  s_tmp = 0
  e_tmp = 0
  for target_class in sorted(class_to_idx.keys()):
    cnt = file_count[target_class]
    e_tmp += cnt
    c_name.append(target_class)
    s_cnt.append(s_tmp)
    e_cnt.append(e_tmp)
    s_tmp += cnt
  
  s_ind = args.fold * shards_count
  e_ind = (args.fold+1) * shards_count
  
  # find the corresponds file. 
  s_idx = 0
  for v in e_cnt:
    if s_ind <= v:
      break
    s_idx += 1

  e_idx = 0
  for v in e_cnt:
    if e_ind < v:
      break
    e_idx += 1
    
  idx_to_class = {i:c for c,i in class_to_idx.items()}
  
  # count set to the start of that specfic file.
  count = s_cnt[s_idx]
  
  if e_idx == len(e_cnt):
    e_idx = e_idx - 1
    
  for i in range(s_idx, e_idx+1):
    
    if 'stack-code' in args.data_dir:
        file_path = idx_to_class[i]
    else:
      file_path = os.path.join(directory, idx_to_class[i])

    print("reading {}, {} : {}".format(file_path, s_ind, e_ind), flush=True)
    annotation_dict = {}
    with tf.io.gfile.GFile(file_path, 'rb') as file:
      with gzip.GzipFile(fileobj=file) as gzip_file:
        for line in gzip_file:
          if count >= s_ind and count < e_ind:
            json_content = json.loads(line.decode('utf-8'))
            item = json_content['id'], json_content['text']
            instances.append(item)
          count += 1
  print("total instance {}".format(len(instances)))
  return instances

if not file_exist:
  print("file not exist, do the preprocessing.")
  if 'stack-code' in args.data_dir:
      files = glob.glob(args.data_dir + '/**/' + '*.json.gz')
  else:
      files = tf.io.gfile.listdir(args.data_dir)

  files.sort()
  files_to_idx = {cls_name: i for i, cls_name in enumerate(files)}
  data = _make_dataset(args, args.data_dir, files_to_idx)

  num_written = 0
  max_len = 0
  with GCSTFRecordWriter(out_fn, auto_close=False) as tfrecord_writer:
    for item in data:  
      filename, text = item
      feature_dict = {
          'id': bytes_feature(filename.encode('utf-8')),
          'text': bytes_feature(text.encode('utf-8')),
      }
      example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
      tfrecord_writer.write(example.SerializeToString())
      num_written += 1
      if num_written % 100 == 0:
        print("Have written {} / {}".format(num_written, len(data)), flush=True)
      
    tfrecord_writer.close()
  print(f'Finished writing {num_written} questions; max len = {max_len}', flush=True)