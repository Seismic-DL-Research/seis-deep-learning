import tensorflow as tf
import my_notebook_modules as mynbm

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_relu"
)
class complex_relu(tf.keras.layers.Layer):
  def __init__(sf, name__='Complex ReLU'):
    super(complex_relu, sf).__init__(name=name__)

  def build(sf, input_shape__):
    pass

  def relu(sf, x__):
    where_negative = tf.where(x__ < 0)
    zeros = tf.zeros((where_negative.shape[0],))
    return tf.tensor_scatter_nd_update(x__, where_negative, zeros)

  def call(sf, inputs__):
    aR, aJ = mynbm.layers.disintegrate_complex(inputs__)

    bR = tf.expand_dims(sf.relu(aR), axis=1)
    bJ = tf.expand_dims(sf.relu(bJ), axis=1)

    return mynbm.layers.integrate_complex(bR, bJ)