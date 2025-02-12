# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Input pipeline for the transformer model to read, filter, and batch examples.

Two things to note in the pipeline:

1. Batching scheme

   The examples encoded in the TFRecord files contain data in the format:
     {"inputs": [variable length array of integers],
      "targets": [variable length array of integers]}
   Where integers in the arrays refer to tokens in the English and German vocab
   file (named `vocab.ende.32768`).

   Prior to batching, elements in the dataset are grouped by length (max between
   "inputs" and "targets" length). Each group is then batched such that:
     group_batch_size * length <= batch_size.

   Another way to view batch_size is the maximum number of tokens in each batch.

   Once batched, each element in the dataset will have the shape:
     {"inputs": [group_batch_size, padded_input_length],
      "targets": [group_batch_size, padded_target_length]}
   Lengths are padded to the longest "inputs" or "targets" sequence in the batch
   (padded_input_length and padded_target_length can be different).

   This batching scheme decreases the fraction of padding tokens per training
   batch, thus improving the training speed significantly.

2. Shuffling

   While training, the dataset is shuffled in two places in the code. The first
   is the list of training files. Second, while reading records using
   `parallel_interleave`, the `sloppy` argument is used to generate randomness
   in the order of the examples.
"""

import os, numpy as np

from absl import logging
import tensorflow as tf

# Buffer size for reading records from a TFRecord file. Each training file is
# 10 MB, so 16 MB allows an entire file to be kept in memory.
READ_RECORD_BUFFER = 16 * 1000 * 1000

# Example grouping constants. Defines length boundaries for each group.
# These values are the defaults used in Tensor2Tensor.
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


def _load_records(filename, buffer_size):
  """Read file and return a dataset of tf.Examples."""
  return tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
  
def parse_example(serialized_example):
  """Return inputs Tensors from a serialized tf.Example."""
  data_fields = {
      "inputs": tf.io.VarLenFeature(tf.int64)
  }
  parsed = tf.io.parse_single_example(serialized_example, data_fields)
  inputs = tf.sparse.to_dense(parsed["inputs"])

  return inputs

def add_bos_eos(x, BOS_id, EOS_id):
    length = tf.size(x)

    # Create a tensor containing <BOS> and <EOS>.
    t = tf.scatter_nd(
      [[0], [length + 1]],
      [BOS_id, EOS_id],
      [length + 2]
      )
    t = tf.cast(t, dtype=x.dtype)
      
    # Fill in the input
    idx = tf.expand_dims(tf.range(1, 1 + length), axis=-1)
    x = tf.tensor_scatter_nd_update(t, idx, x)
    
    return x

def _filter_max_length(example, max_length=256):
  """Indicates whether the example's length is lower than the maximum length."""
  return tf.size(example) <= max_length

def _get_example_length(example):
  """Returns the maximum length between the example inputs and targets."""
  length = tf.maximum(tf.shape(example[0])[0], tf.shape(example[1])[0])
  return length


def _create_min_max_boundaries(max_length,
                               min_boundary=_MIN_BOUNDARY,
                               boundary_scale=_BOUNDARY_SCALE):
  """Create min and max boundary lists up to max_length.

  For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
  returned values will be:
    buckets_min = [0, 4, 8, 16, 24]
    buckets_max = [4, 8, 16, 24, 25]

  Args:
    max_length: The maximum length of example in dataset.
    min_boundary: Minimum length in boundary.
    boundary_scale: Amount to scale consecutive boundaries in the list.

  Returns:
    min and max boundary lists

  """
  # Create bucket boundaries list by scaling the previous boundary or adding 1
  # (to ensure increasing boundary sizes).
  bucket_boundaries = []
  x = min_boundary
  while x < max_length:
    bucket_boundaries.append(x)
    x = max(x + 1, int(x * boundary_scale))

  # Create min and max boundary lists from the initial list.
  buckets_min = [0] + bucket_boundaries
  buckets_max = bucket_boundaries + [max_length + 1]
  return buckets_min, buckets_max


def _batch_examples(dataset, batch_size, max_length):
  """Group examples by similar lengths, and return batched dataset.

  Each batch of similar-length examples are padded to the same length, and may
  have different number of elements in each batch, such that:
    group_batch_size * padded_length <= batch_size.

  This decreases the number of padding tokens per batch, which improves the
  training speed.

  Args:
    dataset: Dataset of unbatched examples.
    batch_size: Max number of tokens per batch of examples.
    max_length: Max number of tokens in an example input or target sequence.

  Returns:
    Dataset of batched examples with similar lengths.
  """
  # Get min and max boundary lists for each example. These are used to calculate
  # the `bucket_id`, which is the index at which:
  # buckets_min[bucket_id] <= len(example) < buckets_max[bucket_id]
  # Note that using both min and max lists improves the performance.
  buckets_min, buckets_max = _create_min_max_boundaries(max_length)

  # Create list of batch sizes for each bucket_id, so that
  # bucket_batch_size[bucket_id] * buckets_max[bucket_id] <= batch_size
  bucket_batch_sizes = [int(batch_size) // x for x in buckets_max]
  # bucket_id will be a tensor, so convert this list to a tensor as well.
  bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

  def example_to_bucket_id(example_input, example_target):
    """Return int64 bucket id for this example, calculated based on length."""
    seq_length = _get_example_length((example_input, example_target))

    # TODO(xunkai): investigate if removing code branching improves performance.
    conditions_c = tf.logical_and(
        tf.less_equal(buckets_min, seq_length), tf.less(seq_length,
                                                        buckets_max))
    bucket_id = tf.reduce_min(tf.where(conditions_c))
    return bucket_id

  def window_size_fn(bucket_id):
    """Return number of examples to be grouped when given a bucket id."""
    return bucket_batch_sizes[bucket_id]

  def batching_fn(bucket_id, grouped_dataset):
    """Batch and add padding to a dataset of elements with similar lengths."""
    bucket_batch_size = window_size_fn(bucket_id)

    # Batch the dataset and add padding so that all input sequences in the
    # examples have the same length, and all target sequences have the same
    # lengths as well. Resulting lengths of inputs and targets can differ.
    return grouped_dataset.padded_batch(bucket_batch_size, ([None], [None]))

  return dataset.apply(
      tf.data.experimental.group_by_window(
          key_func=example_to_bucket_id,
          reduce_func=batching_fn,
          window_size=None,
          window_size_func=window_size_fn))
  
def read_and_batch_from_files(
  file_pattern,
  max_length,
  max_io_parallelism,
  shuffle,
  repeat,
  params=None
  ):
  """Create dataset where each item is a dict of "inputs" and "targets".

  Args:
    file_pattern: String used to match the input TFRecord files.
    batch_size: Maximum number of tokens per global batch of examples.
    max_length: Maximum number of tokens per example
    max_io_parallelism: Max number of cpu cores for parallel input processing.
    shuffle: If true, randomizes order of elements.
    repeat: Number of times to repeat the dataset. If None, the dataset is
      repeated forever.
    static_batch: Whether the batches in the dataset should have static shapes.
      If True, the input is batched so that every batch has the shape
      [batch_size // max_length, max_length]. If False, the input is grouped by
      length, and batched so that batches may have different
      shapes [N, M], where: N * M <= batch_size M <= max_length In general, this
        setting should be False. Dynamic shapes allow the inputs to be grouped
        so that the number of padding tokens is minimized, and helps model
        training. In cases where the input shape must be static (e.g. running on
        TPU), this setting should be set to True.
    num_replicas: Number of GPUs or other workers. We will generate global
      batches, and each global batch is equally divisible by number of replicas.
      Currently it is only effective when static_batch==True. TODO: make it
        effective when static_batch=False.
    ctx: Input context.

  Returns:
    tf.data.Dataset object containing examples loaded from the files.
  """
  dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

  # Read files and interleave results. When training, the order of the examples
  # will be non-deterministic.
  options = tf.data.Options()
  options.experimental_deterministic = False
  dataset = dataset.interleave(
      lambda filename: _load_records(filename, READ_RECORD_BUFFER),
      cycle_length=max_io_parallelism,
      num_parallel_calls=tf.data.experimental.AUTOTUNE).with_options(options)

  # Parse each tf.Example into a dictionary
  # TODO: Look into prefetch_input_elements for performance optimization.  # pylint: disable=g-bad-todo
  dataset = dataset.map(
      parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  BOS_id, EOS_id = params["BOS_id"], params["EOS_id"]
  dataset = dataset.map(
      lambda input: add_bos_eos(input, BOS_id, EOS_id),
      num_parallel_calls=tf.data.experimental.AUTOTUNE
      )

  # Remove examples where the input or target length exceeds the maximum length.
  dataset = dataset.filter(lambda x: _filter_max_length(x, max_length))

  return dataset.repeat(repeat)

def get_training_set(params):
  """Load and return dataset of batched examples for use during training."""
  file_pattern1, file_pattern2 = (
    os.path.join(params["wiki_dir"], "*train*"),
    os.path.join(params["baidu_dir"], "*train*")
  )
  max_length = params["max_length"]
  dataset_X, dataset_Y = (
    read_and_batch_from_files(
      file_pattern1,
      max_length,
      params["max_io_parallelism"],
      shuffle=True,
      repeat=params["repeat_dataset"],
      params=params
      ),
    read_and_batch_from_files(
      file_pattern2,
      max_length,
      params["max_io_parallelism"],
      shuffle=True,
      repeat=params["repeat_dataset"],
      params=params
      )
      )
  dataset = tf.data.Dataset.zip((dataset_X, dataset_Y))

  if params["static_batch"]:
    dataset = dataset.padded_batch(
        # First calculate batch size (token number) per worker, then divide it
        # into sentences, and finally expand to a global batch. It could prove
        # the global batch divisble for distribution strategy.
        int(params["n_tokens"] // params["num_gpus"] // max_length * params["num_gpus"]),
        drop_remainder=True)
  else:
    # Group and batch such that each batch has examples of similar length.
    # TODO(xunkai): _batch_examples might need to do something special for
    # num_replicas.
    dataset = _batch_examples(dataset, params["n_tokens"], max_length)

  # Prefetch the next element to improve speed of input pipeline.
  return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def get_valid_set(params):
  """Load and return dataset of batched examples for use during evaluation."""
  file_pattern1, file_pattern2 = (
    os.path.join(params["wiki_dir"], "*dev*"),
    os.path.join(params["baidu_dir"], "*dev*")
  )
  max_length = params["max_length"]
  dataset_X, dataset_Y = (
    read_and_batch_from_files(
      file_pattern1,
      max_length,
      params["max_io_parallelism"],
      shuffle=False,
      repeat=1,
      params=params
      ),
    read_and_batch_from_files(
      file_pattern2,
      max_length,
      params["max_io_parallelism"],
      shuffle=False,
      repeat=1,
      params=params
      )
  )
  dataset = tf.data.Dataset.zip((dataset_X, dataset_Y))

  if params["static_batch"]:
    dataset = dataset.padded_batch(
        # First calculate batch size (token number) per worker, then divide it
        # into sentences, and finally expand to a global batch. It could prove
        # the global batch divisble for distribution strategy.
        int(params["n_tokens"] // params["num_gpus"] // max_length * params["num_gpus"]),
        drop_remainder=False)
  else:
    # Group and batch such that each batch has examples of similar length.
    # TODO(xunkai): _batch_examples might need to do something special for
    # num_replicas.
    dataset = _batch_examples(dataset, params["n_tokens"], max_length)

  if params["n_valid_steps"] is not None:
    dataset = dataset.take(params["n_valid_steps"])

  # Prefetch the next element to improve speed of input pipeline.
  return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).cache()


def map_data_for_transformer_fn(x, y):
  """Maps data for training, and handles weried behaviors for different vers."""
  # Will transform input x and targets y into tuple(x, y) as new model inputs.
  # For TF v2, the 2nd parameter is omitted to make Keras training work.
  return ((x, y),)
