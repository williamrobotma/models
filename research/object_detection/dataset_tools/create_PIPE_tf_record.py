# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import numpy as np

from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'test set.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'test']
# YEARS = ['VOC2007', 'VOC2012', 'merged']


def pipe_to_tf_example(annotation,
                       img_path,
                       label_map_dict):
  """Convert PIPE landscapes into tf example

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.
  """

  
  image = np.load(img_path)
  # Change to channels last
  image = np.transpose(image, (1, 2, 0))
  # encode as string
  encoded_inputs = image.tostring()
  key = hashlib.sha256(encoded_inputs).hexdigest()


  width = int(image.shape[1])
  height = int(image.shape[0])

  xmin = [annotation[1][0] / width]
  ymin = [annotation[0][0] / height]
  xmax = [annotation[1][1] / width]
  ymax = [annotation[0][1] / height]
  classes = [1]
  classes_text = ['site']
  # truncated = []
  # poses = []
  # difficult_obj = []
  # if 'object' in data:
  #   for obj in data['object']:
  #     difficult = bool(int(obj['difficult']))
  #     if ignore_difficult_instances and difficult:
  #       continue

  #     difficult_obj.append(int(difficult))

  #     xmin.append(float(obj['bndbox']['xmin']) / width)
  #     ymin.append(float(obj['bndbox']['ymin']) / height)
  #     xmax.append(float(obj['bndbox']['xmax']) / width)
  #     ymax.append(float(obj['bndbox']['ymax']) / height)
  #     classes_text.append(obj['name'].encode('utf8'))
  #     classes.append(label_map_dict[obj['name']])
  #     truncated.append(int(obj['truncated']))
  #     poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          os.path.basename(img_path).encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          os.path.basename(img_path).encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_inputs),
      'image/channels': dataset_util.int64_feature(image.shape[-1]),
      'image/format': dataset_util.bytes_feature('np'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      # 'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      # 'image/object/truncated': dataset_util.int64_list_feature(truncated),
      # 'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))

  data_dir = FLAGS.data_dir

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = {item: {'id': 1, 'name': 'site'}}

    logging.info('Reading from PIPE landscapes dataset.')

    annotations_path = os.path.join(data_dir, FLAGS.set + '_coords.json')
    example_paths = glob.glob(os.path.join(data_dir, FLAGS.set,'features','*.npy'))

    examples_list = [os.path.splitext(os.path.basename(example_path))[0] for example_path in example_paths]
    with open(annotations_path, 'r') as f:
      annotations = json.load(f)

    for idx, example in enumerate(examples_list):
      if idx % 100 == 0:
        logging.info('On lanscape %d of %d', idx, len(examples_list))
      path = example_paths[idx]

      annotation = annotations[example]

      tf_example = pipe_to_tf_example(annotation, path, label_map_dict)
      writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()
