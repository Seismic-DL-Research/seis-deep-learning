import tensorflow as tf
import tensorflow.math as tfm

def consec_reduce_max(data__, n__):
  data = data__
  for i in range(1, n__+1):
    data = tfm.reduce_max(data, axis=-i, keepdims=True)
  return data

def consec_reduce_min(data__, n__):
  data = data__
  for i in range(1, n__+1):
    data = tfm.reduce_min(data, axis=-i, keepdims=True)
  return data

def get_stat(data__):
  min = consec_reduce_min(data__, 3)
  max = consec_reduce_max(data__, 3)

  # linearize
  # from N x 2 x 3 x H x W
  # to N x 2 x 1 x 1 x 1

  reshaped = tf.reshape(data__,
                        shape=tuple(tf.shape(data__)[:2])+(1,))

  avg = tfm.reduce_mean(reshaped, axis=-1)
  std = tfm.reduce_std(reshaped, axis=-1)

  return min[:,:,0,0,0], max[:,:,0,0,0], avg, std