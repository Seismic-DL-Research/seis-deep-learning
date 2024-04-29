import tensorflow as tf
import tensorflow.math as tfm
import my_notebook_modules as mynbm

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_batch_norm_2d"
)
class complex_batch_norm_2d(tf.keras.layers.Layer):
  def __init__(sf, pool_size__, activation__=None, name__=None):
    layer_type = tf.constant('cap2d', tf.string)
    layer_name = mynbm.layers.utils.random_name(layer_type)
    super(complex_batch_norm_2d, sf).__init__(name=layer_name.numpy().decode('utf-8'))
    sf.pool_size = pool_size__
    sf.activation = activation__
    sf.universal_strides = [1,1,1,1]

  def build(sf, input_shape__):
    pass

  def call(sf, inputs__):
    aR, aJ = mynbm.layers.utils.disintegrate_complex(inputs__)
    aR_nchw = tf.transpose(aR, perm=[0,1,4,2,3])
    aJ_nchw = tf.transpose(aJ, perm=[0,1,4,2,3])

    shape = tf.shape(aR_nchw)
    dims = [shape[0], shape[1], shape[2], shape[3]*shape[4]]

    flat_matrix_aR_nchw = tf.reshape(aR_nchw, shape=dims)
    flat_matrix_aJ_nchw = tf.reshape(aJ_nchw, shape=dims)

    aR_mean = tfm.reduce_mean(flat_matrix_aR_nchw, axis=-1, keepdims=True)
    aJ_mean = tfm.reduce_mean(flat_matrix_aJ_nchw, axis=-1, keepdims=True)
    aR_std = tfm.reduce_std(flat_matrix_aR_nchw, axis=-1, keepdims=True)
    aJ_std = tfm.reduce_std(flat_matrix_aJ_nchw, axis=-1, keepdims=True)

    aR_mean = tf.expand_dims(aR_mean, axis=-1)
    aJ_mean = tf.expand_dims(aJ_mean, axis=-1)
    aR_std = tf.expand_dims(aR_std, axis=-1)
    aJ_std = tf.expand_dims(aJ_std, axis=-1)



    if sf.activation == None: return end_tensor
    return sf.activation(end_tensor)

  def get_config(sf):
    my_config = super(complex_batch_norm_2d, sf).get_config()
    my_config['activation__'] = sf.activation
    return my_config