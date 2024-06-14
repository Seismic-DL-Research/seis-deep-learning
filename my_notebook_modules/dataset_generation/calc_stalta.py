from numba import jit
import numpy as np

#@njit(fastmath=True)
def stalta(waveform_data, nsta, nlta, trigger):
  # this is a better approach for STA/LTA
  cumsum = np.cumsum(np.abs(waveform_data) ** 2)
  sta = np.concatenate((cumsum[:nsta], cumsum[nsta:] - cumsum[:-nsta]))
  lta = np.concatenate((cumsum[:nlta], cumsum[nlta:] - cumsum[:-nlta]))
  sta[:nlta] = 0
  stalta = sta/lta * (nlta/nsta)
  tp = np.where(stalta > trigger)[0]
  return stalta, tp