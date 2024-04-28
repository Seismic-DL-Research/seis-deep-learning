import tensorflow as tf

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_conv_2d"
)
class complex_conv_2d(tf.keras.layers.Layer):
  def __init__(sf, name__='Complex ReLU'):
    super(complex_conv_2d, sf).__init__(name=name__)

  def build(sf, input_shape__):
    pass

  def relu(sf, x__):
    where_negative = tf.where(x__ < 0)
    zeros = tf.zeros((where_negative.shape[0],))
    return tf.tensor_scatter_nd_update(x__, where_negative, zeros)

  def make_pair(sf, a__, b__):
    a = tf.expand_dims(a__, axis=1)
    b = tf.expand_dims(b__, axis=1)
    return tf.concat([a, b], axis=1)

  def call(sf, inputs__):
    aR = inputs__[:,0]
    aJ = inputs__[:,1]
    bR = tf.expand_dims(sf.relu(aR), axis=1)
    bJ = tf.expand_dims(sf.relu(bJ), axis=1)

    return 