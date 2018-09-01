from typing import Tuple, Iterator

import numpy as np
import tensorflow as tf


class Dataset(object):
  def __init__(self):
    self._batch_size = 16
    self._time_seq_length = 32
    self._x_dim = 8
    self._y_dim = 1

  def input_fn(self) -> tf.data.Dataset:
    def _gen() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
      while True:
        yield self._batch()

    dataset = tf.data.Dataset.from_generator(
      _gen,
      (tf.float32, tf.float32),
      ((self._batch_size, self._time_seq_length, self._x_dim), (self._batch_size, self._y_dim))
    ).map(
      lambda x_batch, y_batch: ({"x": x_batch}, y_batch)
    )

    return dataset

  def _batch(self) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.random_sample((self._batch_size, self._time_seq_length, self._x_dim))
    y = np.random.random_sample((self._batch_size, self._y_dim))
    return x, y
