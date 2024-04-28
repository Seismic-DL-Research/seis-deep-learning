import tensorflow as tf
import my_notebook_modules as mynbm

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_avg_pool_2d"
)
class complex_avg_pool_2d(tf.keras.layers.Layer):
  def __init__(sf, pool_size__, name__='Complex Avg Pool 2D'):
    super(complex_avg_pool_2d, sf).__init__(name=name__)
    sf.pool_size = pool_size__
    sf.universal_strides = [1,1,1,1]

  def build(sf, input_shape__):
    pass

  def call(sf, inputs__):
    u, v = mynbm.layers.disintegrate_complex(inputs__)
    
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

    return mynbm.layers.integrate_complex(pool_u, pool_v)