import tensorflow as tf
import tensorflow.math as tfm
from scipy.signal import stft

@tf.py_function(Tout=tf.int32)
def get_year(x):
  year = int(tf.strings.split(input=x, sep='-')[0])
  return year

@tf.py_function(Tout=tf.float32)
def stft_process(x, nperseg__, noverlap__):
  f, t, Z = stft(x, fs=100, nperseg=nperseg__, noverlap=noverlap__)
  ZR = tf.expand_dims(tfm.real(Z), axis=1)
  ZJ = tf.expand_dims(tfm.imag(Z), axis=1)
  Z = tf.concat([ZR, ZJ], axis=1)
  return Z

@tf.py_function(Tout=tf.float32)
def consec_reduce_max(data__, n__):
  data = data__
  for i in range(1, n__+1):
    data = tf.reduce_max(data, axis=-i, keepdims=True)
  return data

# @tf.py_function(Tout=tf.float32)
# def consec_reduce_min(data__, n__):
#   data = data__
#   for i in range(1, n__+1):
#     data = tf.reduce_min(data, axis=-i, keepdims=True)
#   return data

@tf.py_function(Tout=tf.float32)
def get_maxavg(data__):
  maxv = consec_reduce_max(data__, 3)[0,0,0]
  avgv = tfm.reduce_mean(tf.reshape(data__, shape=(-1,)))
  return tf.concat([maxv, avgv], axis=0)

@tf.py_function(Tout=tf.float32)
def get_imag_real_part(data__):
  return tf.concat([data__[:,0], data__[:,1]], axis=0)
