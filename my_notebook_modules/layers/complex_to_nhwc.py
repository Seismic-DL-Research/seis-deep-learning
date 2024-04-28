import tensorflow as tf
import tensorflow.math as tfm
import my_notebook_modules as mynbm 

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_to_nhwc"
)
class complex_to_nhwc(tf.keras.layers.Layer):
  def __init__(sf):
    layer_type = tf.constant('cinpt', tf.string)
    layer_name = mynbm.layers.utils.random_name(layer_type)
    super(complex_to_nhwc, sf).__init__(name=layer_name.numpy().decode('utf-8'))

  def build(sf, input_shape__):
    pass

  def call(sf, input__):
    # input__: N x C x 2 x H x W
    nhwc_tensor = tf.transpose(input__, perm=[0,2,3,4,1])
    return nhwc_tensor