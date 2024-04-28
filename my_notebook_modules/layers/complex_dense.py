import tensorflow as tf
import my_notebook_modules as mynbm

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_dense"
)
class complex_dense(tf.keras.layers.Layer):
  def __init__(sf, dense_unit__, activation__=None):
    layer_type = tf.constant('cdns', tf.string)
    layer_name = mynbm.layers.utils.random_name(layer_type)
    super(complex_dense, sf).__init__(name=layer_name.numpy().decode('utf-8'))
    sf.dense_unit = dense_unit__
    sf.activation = activation__

  def build(sf, input_shape__):
    # input shape: N x 2 x D
    sf.WR = sf.add_weight(
          shape=(input_shape__[-1], sf.dense_unit),
          initializer=tf.keras.initializers.GlorotNormal(),
          trainable=True,
          name='complex_dense_QR'
      )
    sf.WJ = sf.add_weight(
          shape=(input_shape__[-1], sf.dense_unit),
          initializer=tf.keras.initializers.GlorotNormal(),
          trainable=True,
          name='complex_dense_QJ'
      )
    sf.BR = sf.add_weight(
          shape=(1, sf.dense_unit),
          initializer=tf.keras.initializers.GlorotNormal(),
          trainable=True,
          name='complex_dense_BR'
      )
    sf.BJ = sf.add_weight(
          shape=(1, sf.dense_unit),
          initializer=tf.keras.initializers.GlorotNormal(),
          trainable=True,
          name='complex_dense_BJ'
      )
    pass

  def call(sf, inputs__):
    # aR: N x D
    aR, aJ = mynbm.layers.utils.disintegrate_complex(inputs__)

    # bR: N x D'
    bR = tf.matmul(aR, sf.WR) + sf.BR
    bJ = tf.matmul(aJ, sf.WJ) + sf.BJ

    end_tensor = mynbm.layers.utils.integrate_complex(bR, bJ)
    if sf.activation == None: return end_tensor
    return sf.activation(end_tensor)
    
  def get_config(sf):
    my_config = super(complex_dense, sf).get_config()
    my_config['activation__'] = sf.activation
    return my_config
    