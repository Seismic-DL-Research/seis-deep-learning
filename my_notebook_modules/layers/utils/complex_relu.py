import tensorflow as tf
import my_notebook_modules as mynbm

@tf.function
def relu_computation(x__):
  where_negative = tf.where(x__ < 0)
  zeros = tf.zeros((tf.shape(where_negative)[0],))
  return tf.tensor_scatter_nd_update(x__, where_negative, zeros)

def complex_relu():
  @tf.function
  def core_opt(inputs__):
    aR, aJ = mynbm.layers.utils.disintegrate_complex(inputs__)
    bR = relu_computation(aR)
    bJ = relu_computation(aJ)

    end_tensor = mynbm.layers.utils.integrate_complex(bR, bJ)
    return end_tensor
  return core_opt