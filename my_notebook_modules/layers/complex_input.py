import tensorflow as tf
import tensorflow.math as tfm

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_input"
)
class complex_input(tf.keras.layers.Layer):
  def __init__(sf):
    super(complex_input, sf).__init__()

  def build(sf, input_shape__):
    pass

  def call(sf, input__):
    R = input__[0]
    J = input__[1]
    return R, J