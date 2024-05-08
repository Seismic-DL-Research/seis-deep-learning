import tensorflow as tf
import my_notebook_modules as mynbm

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_conv_2d"
)
class complex_conv_2d_transpose(tf.keras.layers.Layer):
  def __init__(sf, kernel_size__, kernel_total__, padding__='VALID',
              activation__=None):
    layer_type = tf.constant('cc2d', tf.string)
    layer_name = mynbm.layers.utils.random_name(layer_type)
    super(complex_conv_2d_transpose, sf).__init__(name=layer_name.numpy().decode('utf-8'))
    sf.kernel_size = kernel_size__
    sf.kernel_total = kernel_total__
    sf.activation = activation__
    sf.padding = padding__
    sf.universal_strides = [1,1,1,1]

  def build(sf, input_shape__):
    # input shape: N x 2 x H x W x C
    sf.kernel_p = sf.add_weight(
        shape=sf.kernel_size + (sf.kernel_total, input_shape__[-1]),
        initializer=tf.keras.initializers.GlorotUniform(),
        trainable=True,
        name='kernel_p'
      )
    sf.kernel_q = sf.add_weight(
        shape=sf.kernel_size + ( sf.kernel_total, input_shape__[-1]),
        initializer=tf.keras.initializers.GlorotUniform(),
        trainable=True,
        name='kernel_q'
      )
    sf.bias_p = sf.add_weight(
        shape=(sf.kernel_total,),
        initializer=tf.keras.initializers.GlorotUniform(),
        trainable=True,
        name='bias_p'
      )
    sf.bias_q = sf.add_weight(
        shape=(sf.kernel_total,),
        initializer=tf.keras.initializers.GlorotUniform(),
        trainable=True,
        name='bias_q'
      )
    pass

  def call(sf, inputs__):
    u, v = mynbm.layers.utils.disintegrate_complex(inputs__)
    
    convtr_up = tf.nn.conv2d_transpose(
        input=u, filters=sf.kernel_p,
        strides=sf.universal_strides,
        output_shape=[2,tf.shape(u)[1] + sf.kernel_size[0], tf.shape(u)[2] + sf.kernel_size[1], sf.kernel_total],
        padding=sf.padding
    )
    convtr_vq = tf.nn.conv2d_transpose(
        input=u, filters=sf.kernel_p,
        strides=sf.universal_strides,
        output_shape=[2,tf.shape(u)[1] + sf.kernel_size[0], tf.shape(u)[2] + sf.kernel_size[1], sf.kernel_total],
        padding=sf.padding
    )
    convtr_uq = tf.nn.conv2d_transpose(
        input=u, filters=sf.kernel_p,
        strides=sf.universal_strides,
        output_shape=[2,tf.shape(u)[1] + sf.kernel_size[0], tf.shape(u)[2] + sf.kernel_size[1], sf.kernel_total],
        padding=sf.padding
    )
    convtr_vp = tf.nn.conv2d_transpose(
        input=u, filters=sf.kernel_p,
        strides=sf.universal_strides,
        output_shape=[2,tf.shape(u)[1] + sf.kernel_size[0], tf.shape(u)[2] + sf.kernel_size[1], sf.kernel_total],
        padding=sf.padding
    )

    real_conv = convtr_up + convtr_vq
    imag_conv = convtr_uq + convtr_vp

    end_tensor = mynbm.layers.utils.integrate_complex(real_conv, imag_conv)
    if sf.activation == None: return end_tensor
    return sf.activation(end_tensor)
    
  def get_config(sf):
    my_config = super(complex_conv_2d_transpose, sf).get_config()
    my_config['activation__'] = sf.activation
    return my_config
    