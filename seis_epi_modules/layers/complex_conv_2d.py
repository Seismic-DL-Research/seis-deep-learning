import tensorflow as tf
import tensorflow.math as tfm
import tensorflow.linalg as tfl

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_conv_2d"
)
class complex_conv_2d(tf.keras.layers.Layer):
  def __init__(sf, kernel_size__, kernel_total__):
    super(complex_conv_2d, sf).__init__()
    sf.kernel_size = kernel_size__
    sf.kernel_total = kernel_total__
    sf.universal_strides = [1,1,1,1]

  def build(sf, input_shape__):
    sf.kernel_p = sf.add_weight(
          shape=sf.kernel_size + (3, sf.kernel_total),
          initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
          trainable=True,
          name='kernel_p'
      )
    sf.kernel_q = sf.add_weight(
          shape=sf.kernel_size + (3, sf.kernel_total),
          initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
          trainable=True,
          name='kernel_q'
      )
    pass

  def call(sf, inputs__):
    u = tfm.real(inputs__)
    v = tfm.imag(inputs__)
    conv_up = tf.nn.conv2d(
        input=u, filters=sf.kernel_p,
        strides=sf.universal_strides,
        padding='VALID'
    )
    conv_vq = tf.nn.conv2d(
        input=v, filters=sf.kernel_q,
        strides=sf.universal_strides,
        padding='VALID'
    )
    conv_uq = tf.nn.conv2d(
        input=u, filters=sf.kernel_q,
        strides=sf.universal_strides,
        padding='VALID'
    )
    conv_vp = tf.nn.conv2d(
        input=v, filters=sf.kernel_p,
        strides=sf.universal_strides,
        padding='VALID'
    )

    real_conv = conv_up + conv_vq
    imag_conv = conv_uq + conv_vp
    return tf.complex(real_conv, imag_conv)