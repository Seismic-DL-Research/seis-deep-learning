import tensorflow as tf
import my_notebook_modules as mynbm

@tf.function
def leaky_relu_computation(x__, a__):
  where_negative = tf.where(x__ < 0)
  masked_values = tf.boolean_mask(x__, x__ < 0) * a__
  return tf.tensor_scatter_nd_update(x__, where_negative, masked_values)

def complex_leaky_relu(a__):
  @tf.function
  def core_opt(inputs__):
    aR, aJ = mynbm.layers.utils.disintegrate_complex(inputs__)
    bR = leaky_relu_computation(aR, a__)
    bJ = leaky_relu_computation(aJ, a__)

    end_tensor = mynbm.layers.utils.integrate_complex(bR, bJ)
    return end_tensor
  return core_opt