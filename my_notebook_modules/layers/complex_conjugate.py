import tensorflow as tf
import tensorflow.math as tfm
import my_notebook_modules as mynbm

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_conjugate"
)
class complex_conjugate(tf.keras.layers.Layer):
  def __init__(sf, activation__):
    layer_type = tf.constant('ccon', tf.string)
    layer_name = mynbm.layers.utils.random_name(layer_type)
    super(complex_conjugate, sf).__init__(name=layer_name.numpy().decode('utf-8'))
    sf.activation = activation__

  def build(sf, input_shape__):
    pass

  def call(sf, inputs__):
    aR, aJ = mynbm.layers.utils.disintegrate_complex(inputs__)

    aR2 = tfm.square(aR)
    aJ2 = tfm.square(aJ)
    end_tensor = tfm.sqrt(aR2 + aJ2)

    if sf.activation == None: return end_tensor
    return sf.activation(end_tensor)
    
  def get_config(sf):
    my_config = super(complex_conjugate, sf).get_config()
    my_config['activation__'] = sf.activation
    return my_config
    