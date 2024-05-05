import tensorflow as tf
import tensorflow.math as tfm
import my_notebook_modules as mynbm

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

def filterFunc_specific_key_range(bot_range__, top_range__, key__):
  def core_opt(x):
    return tfm.logical_and(x[key__] > bot_range__,
                           x[key__] < top_range__)
  return core_opt

def mapFunc_clip(a, b):
  def core_opt(x):
    x['data'] = x['data'][:,a:b]
    return x
  return core_opt

# using tf.py_function as a symbolic tensor can't be loaded
# and processed further directly by tf.strings.split

def imprudent_mapFunc_add_year(x):
  year = mynbm.dataset_utils.op.get_year(x['start'])
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

def mapFunc_stft(nperseg__, noverlap__, window__, clip_freq_index__,
                 clip_time_index__):
  def core_opt(x):
    Z = mynbm.dataset_utils.op.stft_process(x['data'], nperseg__, noverlap__,
                                            window__)
    if clip_freq_index__ == 0:
      x['data'] = Z[:,:,:,:clip_time_index__]
    else:
      x['data'] = Z[:,:,:clip_freq_index__,:clip_time_index__]
    return x
  return core_opt

def filterFunc_reject_outliers(bot_range__, top_range__):
  def core_opt(x):
    minmax = mynbm.dataset_utils.op.tfpy_func.get_minmax(x['data'])
    minv = minmax[0]
    maxv = minmax[1]
    cond_real = tfm.logical_and(minv[0] > bot_range__, maxv[0] < top_range__)
    cond_imag = tfm.logical_and(minv[1] > bot_range__, maxv[1] < top_range__)
    return tfm.logical_and(cond_real, cond_imag)
  return core_opt
