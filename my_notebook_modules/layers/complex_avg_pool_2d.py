import tensorflow as tf

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_avg_pool_2d"
)
class complex_avg_pool_2d(tf.keras.layers.Layer):
  def __init__(sf, pool_size__):
    super(complex_avg_pool_2d, sf).__init__()
    sf.pool_size = pool_size__
    sf.universal_strides = [1,1,1,1]

  def build(sf, input_shape__):
    pass

  def make_pair(sf, a__, b__):
    a = tf.expand_dims(a__, axis=1)
    b = tf.expand_dims(b__, axis=1)
    return tf.concat([a, b], axis=1)

  def call(sf, inputs__):
    u = inputs__[:,0]
    v = inputs__[:,1]
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

    return sf.make_pair(pool_u, pool_v)