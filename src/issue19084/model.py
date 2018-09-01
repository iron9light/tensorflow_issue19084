import tensorflow as tf


class Model(object):
  def __init__(self):
    self._rnn_init_state_trainable = True

  def apply(self, x: tf.Tensor, training: bool) -> tf.Tensor:
    cell = tf.nn.rnn_cell.GRUCell(num_units=32)
    with tf.name_scope("rnn1"):
      rnn_output1, _ = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=x,
        dtype=tf.float32,
        initial_state=get_initial_cell_state(cell, x) if self._rnn_init_state_trainable else None,
      )
    with tf.name_scope("rnn2"):
      rnn_output2, _ = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=x,
        dtype=tf.float32,
        initial_state=get_initial_cell_state(cell, x) if self._rnn_init_state_trainable else None,
      )

    rnn_output = tf.concat([rnn_output1[:, -1], rnn_output2[:, -1]], axis=-1)

    return tf.layers.dense(rnn_output, 1)


class InitStateVarLayer(tf.layers.Layer):
  def __init__(self, name, batch_size, state_size, dtype=None):
    super(InitStateVarLayer, self).__init__(name=name, dtype=dtype)
    self._batch_size = batch_size
    self._state_size = state_size

  def build(self, _):
    from tensorflow.python.ops.rnn_cell_impl import _concat
    c = _concat(1, self._state_size, static=True)
    # size = self.add_variable("init_state", shape=c, initializer=tf.initializers.zeros)
    size = self.add_weight(name="init_state", shape=c, dtype=self.dtype, initializer=tf.initializers.zeros,
                           trainable=True)
    self._size = tf.tile(size, [self._batch_size] + [1] * (len(c) - 1))
    self.built = True

  def call(self, *args, **kwargs):
    return self._size

  def compute_output_shape(self, input_shape):
    return tf.TensorShape([self._batch_size, self._state_size])


def get_initial_cell_state(cell, inputs, dtype=None):
  # tf.train.get_or_create_global_step()
  state_size = cell.state_size
  batch_size = inputs.shape[0].value
  i = 0
  # with tf.name_scope("StateVar", values=[batch_size]):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""

  def get_state_shape(s):
    """Combine s with batch_size to get a proper tensor shape."""
    nonlocal i
    name = "init_state_" + str(i)
    i = i + 1
    # c = _concat(1, s, static=True)
    # size = tf.get_variable(name, shape=c, dtype=dtype, initializer=tf.initializers.zeros)
    # size = tf.tile(size, [batch_size] + [1] * (len(c) - 1))
    size = InitStateVarLayer(name, batch_size, s, dtype=dtype).apply(inputs)
    return size

  from tensorflow.python.util import nest
  return nest.map_structure(get_state_shape, state_size)
