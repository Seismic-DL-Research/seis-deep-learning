import tensorflow as tf
import tensorflow.math as tfm
import tensorflow.linalg as tfl

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="ComplexAvgPool2D"
)
class ComplexAvgPool2D(tf.keras.layers.Layer):
  def __init__(sf, pool_size__):
    super(ComplexAvgPool2D, sf).__init__()
    sf.pool_size = pool_size__
    sf.universal_strides = [1,1,1,1]

  def build(sf, input_shape__):
    pass

  def call(sf, inputs__):
    u = tfm.real(inputs__)
    v = tfm.imag(inputs__)
    pool_u = tf.nn.avg_pool2d(
        input=u, ksize=sf.pool_size,
        strides=sf.universal_strides,
        padding='VALID'
    )
    pool_v = tf.nn.avg_pool2d(
        input=v, ksize=sf.pool_size,
        strides=sf.universal_strides,
        padding='VALID'
    )
    return tf.complex(pool_u, pool_v)