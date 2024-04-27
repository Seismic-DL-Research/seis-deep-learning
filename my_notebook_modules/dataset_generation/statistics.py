from numba import njit
import numpy as np

@njit(fastmath=True)
def power(data__, w__):
  cumsum = np.cumsum(data__ ** 2)
  power = np.concatenate((cumsum[w__:] - cumsum[:-w__], cumsum[:w__]))
  return power

@njit(fastmath=True)
def stalta(waveform_data__, preprocessed_data__,
                   nsta__, nlta__, stalta_trigger__,
                   data_trigger__, normalize__=True):
  cumsum = np.cumsum(np.abs(waveform_data__) ** 2)
  sta = np.concatenate((cumsum[:nsta__],
                        cumsum[nsta__:] - cumsum[:-nsta__]))
  lta = np.concatenate((cumsum[:nlta__],
                        cumsum[nlta__:] - cumsum[:-nlta__])) + 1e-9
  sta[:nlta__] = 0
  stalta = sta/lta * (nlta__/nsta__)
  if normalize__: stalta /= np.max(stalta)

  tp, avg_ratio, score = -1, -1, -1

  try:
    notable_data = np.argwhere(preprocessed_data__ > data_trigger__)[:,0]
    trigger_indices = np.argwhere(stalta[notable_data] > stalta_trigger__)[:,0]
    tp = notable_data[trigger_indices][0]

    if tp < 400 or tp + 6000 > 12000:
      raise Exception('procerror')

    pre_aavg = np.average(np.abs(waveform_data__[tp-100:tp]))
    pos_aavg = np.average(np.abs(waveform_data__[tp:tp+250]))

    pre = np.average(np.abs(waveform_data__ ** 2)[tp-25:tp])
    pos = np.average(np.abs(waveform_data__ ** 2)[tp:tp+25])

    score = pre/pos
    aavg_ratio = pre_aavg/pos_aavg
  except:
    tp = -1
    pass

  return stalta, tp, sta, lta, aavg_ratio, score