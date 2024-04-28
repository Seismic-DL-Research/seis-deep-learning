import tensorflow as tf
import my_notebook_modules as mynbm

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
    aR_shape = tf.shape(aR)
    flattened_size = aR_shape[1] * aR_shape[2] * aR_shape[3]
    aR = tf.reshape(aR, (tf.shape(aR)[0], flattened_size))
    aJ = tf.reshape(aJ, (tf.shape(aJ)[0], flattened_size))


    return mynbm.layers.utils.integrate_complex(aR, aJ)
  
    