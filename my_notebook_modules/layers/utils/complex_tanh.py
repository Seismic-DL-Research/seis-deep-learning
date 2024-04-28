import tensorflow as tf
import tensorflow.math as tfm
import my_notebook_modules as mynbm

@tf.function
def tanh_computation(x__, A__, k__, b__):
  return tfm.tanh(A__* x__ + b__)

@tf.function
def complex_tanh(A__, k__, b__):
  def core_opt(inputs__):
    aR, aJ = mynbm.layers.utils.disintegrate_complex(inputs__)
    bR = tanh_computation(aR, A__, k__, b__)
    bJ = tanh_computation(aJ, A__, k__, b__)

    end_tensor = mynbm.layers.utils.integrate_complex(bR, bJ)
    return end_tensor
  return core_opt