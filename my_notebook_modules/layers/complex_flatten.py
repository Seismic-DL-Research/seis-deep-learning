import tensorflow as tf
import my_notebook_modules as mynbm
import uuid

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_flatten"
)
class complex_flatten(tf.keras.layers.Layer):
  def __init__(sf):
    layer_type = tf.constant('fltn', tf.string)
    layer_name = mynbm.layers.utils.random_name(layer_type)
    super(complex_flatten, sf).__init__(name=layer_name.numpy().decode('utf-8'))

  def build(sf, input_shape__):
    pass

  def call(sf, inputs__):
    aR, aJ = mynbm.layers.utils.disintegrate_complex(inputs__)

    aR = tf.reshape(aR, (tf.shape(aR)[0], -1))
    aJ = tf.reshape(aJ, (tf.shape(aJ)[0], -1))

    return mynbm.layers.utils.integrate_complex(aR, aJ)
  
    