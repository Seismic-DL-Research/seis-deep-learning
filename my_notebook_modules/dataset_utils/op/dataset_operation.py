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

def mapFunc_stft(nperseg__, noverlap__, clip_freq_index__):
  def core_opt(x):
    Z = mynbm.dataset_utils.op.stft_process(x['data'], nperseg__, noverlap__)
    x['data'] = Z[:,:,:clip_freq_index__,:]
    return x
  return core_opt

def filterFunc_reject_outliers(max__, avg__):
  def core_opt(x):
    real_imag = mynbm.dataset_utils.op.get_imag_real_part(x['data'])
    real_part = real_imag[0]
    imag_part = real_imag[1]
    
    # real part test
    maxavg = mynbm.dataset_utils.op.get_maxavg(real_part)
    maxv = maxavg[0]
    avgv = maxavg[1]
    if maxv > max__: return False
    if avgv > avg__: return False

    # imag part test
    maxavg = mynbm.dataset_utils.op.get_maxavg(imag_part)
    maxv = maxavg[0]
    avgv = maxavg[1]
    if maxv > max__: return False
    if avgv > avg__: return False
    return True
  return core_opt
