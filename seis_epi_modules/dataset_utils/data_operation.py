import tensorflow as tf
import tensorflow.math as tfm

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
  year = tf.py_function(get_year, inp=[x['start']], Tout=tf.int32)
  x['year'] = year
  return x