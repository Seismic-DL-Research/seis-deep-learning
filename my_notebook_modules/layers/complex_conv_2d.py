import tensorflow as tf

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_conv_2d"
)
class complex_conv_2d(tf.keras.layers.Layer):
  def __init__(sf, kernel_size__, kernel_total__, 
               name__='Complex Convolution 2D'):
    super(complex_conv_2d, sf).__init__(name=name__)
    sf.kernel_size = kernel_size__
    sf.kernel_total = kernel_total__
    sf.universal_strides = [1,1,1,1]

  def build(sf, input_shape__):
    # input shape: N x 2 x H x W x C
    sf.kernel_p = sf.add_weight(
          shape=sf.kernel_size + (input_shape__[-1], sf.kernel_total),
          initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
          trainable=True,
          name='kernel_p'
      )
    sf.kernel_q = sf.add_weight(
          shape=sf.kernel_size + (input_shape__[-1], sf.kernel_total),
          initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
          trainable=True,
          name='kernel_q'
      )
    pass

  def make_pair(sf, a__, b__):
    a = tf.expand_dims(a__, axis=1)
    b = tf.expand_dims(b__, axis=1)
    return tf.concat([a, b], axis=1)

  def call(sf, inputs__):
    u = inputs__[:,0]
    v = inputs__[:,1]
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

    return sf.make_pair(real_conv, imag_conv)