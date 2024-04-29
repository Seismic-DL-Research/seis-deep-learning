import tensorflow as tf
import my_notebook_modules as mynbm

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_max_pool_2d"
)
class complex_max_pool_2d(tf.keras.layers.Layer):
  def __init__(sf, pool_size__, activation__=None, name__=None):
    layer_type = tf.constant('cmp2d', tf.string)
    layer_name = mynbm.layers.utils.random_name(layer_type)
    super(complex_max_pool_2d, sf).__init__(name=layer_name.numpy().decode('utf-8'))
    sf.pool_size = pool_size__
    sf.activation = activation__
    sf.universal_strides = [1,1,1,1]

  def build(sf, input_shape__):
    pass

  def call(sf, inputs__):
    u, v = mynbm.layers.utils.disintegrate_complex(inputs__)
    
    pool_u = tf.nn.max_pool2d(
        input=u, ksize=sf.pool_size,
        strides=sf.universal_strides,
        padding='VALID'
    )
    pool_v = tf.nn.max_pool2d(
        input=v, ksize=sf.pool_size,
        strides=sf.universal_strides,
        padding='VALID'
    )
    
    end_tensor = mynbm.layers.utils.integrate_complex(pool_u, pool_v)
    if sf.activation == None: return end_tensor
    return sf.activation(end_tensor)

  def get_config(sf):
    my_config = super(complex_max_pool_2d, sf).get_config()
    my_config['activation__'] = sf.activation
    return my_config