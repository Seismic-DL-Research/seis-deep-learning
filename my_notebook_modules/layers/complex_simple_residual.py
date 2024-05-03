import tensorflow as tf
import tensorflow.linalg as tfl
import my_notebook_modules as mynbm

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_residual_2d"
)
class complex_simple_residual(tf.keras.layers.Layer):
  def __init__(sf, activation__=None):
    layer_type = tf.constant('cr2d', tf.string)
    layer_name = mynbm.layers.utils.random_name(layer_type)
    super(complex_simple_residual, sf).__init__(name=layer_name.numpy().decode('utf-8'))
    sf.activation = activation__

  def build(sf, input_shape__):
    pass

  def call(sf, input__):
    alphaR, alphaJ = mynbm.layers.utils.disintegrate_complex(input__[1])
    betaR, betaJ = mynbm.layers.utils.disintegrate_complex(input__[0])

    R = alphaR + betaR
    J = alphaJ + betaJ
    end_tensor = mynbm.layers.utils.integrate_complex(R, J)
    return sf.activation(end_tensor)

  def get_config(sf):
    my_config = super(complex_simple_residual, sf).get_config()
    my_config['activation__'] = sf.activation
    return my_config