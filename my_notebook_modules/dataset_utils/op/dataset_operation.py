import tensorflow as tf
import tensorflow.math as tfm
from scipy.signal import stft

def mapFunc_normalize(x):
  maxVal = tf.expand_dims(tfm.reduce_max(x['data'], axis=-1), axis=-1)
  x['data'] = x['data'] / maxVal
  return x

def filterFunc_metadata(aavg_ratio, dist, magn):
  def core_opt(x):
    return tfm.logical_and(
        tfm.logical_and(tfm.logical_and(x['aavg_ratio'] < aavg_ratio[1],
                                        x['aavg_ratio'] > aavg_ratio[0]),
                        x['dist'] < dist),
        x['magn'] > magn)
  return core_opt

def mapFunc_clip(a, b):
  def core_opt(x):
    x['data'] = x['data'][:,a:b]
    return x
  return core_opt

# using tf.py_function as a symbolic tensor can't be loaded
# and processed further directly by tf.strings.split

@tf.py_function(Tout=tf.int32)
def get_year(x):
  year = int(tf.strings.split(input=x, sep='-')[0])
  return year

def imprudent_mapFunc_add_year(x):
  year = get_year(x['start'])
  x['year'] = year
  return x

def filterFunc_split_by_year(condition, splitAt):
  def core_opt(x):
    if condition == 'before':
      return x['year'] < splitAt
    elif condition == 'after':
      return x['year'] > splitAt
    else:
      return x
  return core_opt

@tf.py_function(Tout=tf.float32)
def stft_process(x, nperseg__, noverlap__):
  f, t, Z = stft(x, fs=100, nperseg=nperseg__, noverlap=noverlap__)
  ZR = tf.expand_dims(tfm.real(Z), axis=1)
  ZJ = tf.expand_dims(tfm.imag(Z), axis=1)
  Z = tf.concat([ZR, ZJ], axis=1)
  return Z

def mapFunc_stft(nperseg__, noverlap__):
  def core_opt(x):
    Z = stft_process(x['data'], nperseg__, noverlap__)
    x['data'] = Z
    return x
  return core_opt

