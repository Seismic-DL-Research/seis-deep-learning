import tensorflow as tf
import my_notebook_modules as mynbm
import uuid

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_conv_2d"
)
class complex_conv_2d(tf.keras.layers.Layer):
  def __init__(sf, kernel_size__, kernel_total__,
              activation__):
    layer_type = tf.constant('cc2d', tf.string)
    layer_name = mynbm.layers.utils.random_name(layer_type)
    super(complex_conv_2d, sf).__init__(name=layer_name.numpy().decode('utf-8'))
    sf.kernel_size = kernel_size__
    sf.kernel_total = kernel_total__
    sf.activation = activation__
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

  def call(sf, inputs__):
    u, v = mynbm.layers.utils.disintegrate_complex(inputs__)
    
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

    # end_tensor: N x 2 x H x W x C
    end_tensor = mynbm.layers.utils.integrate_complex(real_conv, imag_conv)
    return sf.activation(end_tensor)
    
  def get_config(sf):
    my_config = super(complex_conv_2d, sf).get_config()
    my_config['activation__'] = sf.activation
    return my_config
    