import tensorflow as tf
import tensorflow.math as tfm
import scipy
from scipy.signal import stft

@tf.py_function(Tout=tf.int32)
def get_year(x):
  year = int(tf.strings.split(input=x, sep='-')[0])
  return year

@tf.py_function(Tout=tf.float32)
def stft_process(x, nperseg__, noverlap__, window__):
  window = window__.numpy().decode('utf-8')
  f, t, Z = stft(x, fs=100, nperseg=nperseg__, noverlap=noverlap__,
                 window=scipy.signal.get_window(window, nperseg__))
  ZR = tf.expand_dims(tfm.real(Z), axis=0)
  ZJ = tf.expand_dims(tfm.imag(Z), axis=0)
  Z = tf.concat([ZR, ZJ], axis=0)
  return Z

@tf.py_function(Tout=tf.float32)
def consec_reduce_max(data__, n__):
  n = int(n__)
  data = data__
  for i in range(1, n+1):
    data = tf.reduce_max(data, axis=-i, keepdims=True)
  return data

@tf.py_function(Tout=tf.float32)
def consec_reduce_min(data__, n__):
  data = data__
  for i in range(1, n__+1):
    data = tf.reduce_min(data, axis=-i, keepdims=True)
  return data

@tf.py_function(Tout=tf.float32)
def get_minmax(data__):
  # data: 2 x 3 x 50 x 51
  minv = consec_reduce_min(data__, 3)[:,0,0,0]
  maxv = consec_reduce_max(data__, 3)[:,0,0,0]

  # minv/maxv: 2
  return tf.convert_to_tensor([minv, maxv])
