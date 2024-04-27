import obspy
import my_notebook_modules as mynbm
import numpy as np

def clip_event(seed_name__, temp_folder__, power_windows__, nsta__,
               nlta__, trigger_threshold__, data_threshold__):
  modes = ['UD', 'EW', 'NS']

  # waveforms, aavgratio, ovrratio, magn, evla, evlo, stla,
  # stlo, dist, filename, start, end, year
  info_returned = [[0]*3] + [0]*12
  tp = -1
  overall_ratio = -1
  for i in range(3):
    file_location = f'{modes[i]}/{temp_folder__}/{seed_name__}.{modes[i]}1'
    stream = obspy.read(file_location).detrend(type='constant')
    waveform = stream[0].data * stream[0].stats.calib * 100
    if (i == 0):
      squared_waveform = waveform ** 2
      
      tp, aavg_ratio, _ = mynbm.dataset_generation.decisor(
        waveform/np.max(waveform), power_windows__,
        nsta__, nlta__, trigger_threshold__,
        data_threshold__
      )

      if tp == -1: return -1

      overall_ratio = (np.average(squared_waveform[:tp])/
                      np.average(squared_waveform[tp:]))
                      
      if aavg_ratio > 0.5 or aavg_ratio < 0: return -1
      if overall_ratio > 0.5: return -1
      if len(waveform[tp-400:tp+6000]) != (6000+400): return -1
    info_returned[0][i] = waveform[tp-400:tp+6000]

  knet_stats = stream[0].stats.knet
  stat_keys = ['mag', 'evla', 'evlo', 'stla', 'stlo']

  info_returned[1] = aavg_ratio
  info_returned[2] = overall_ratio
  for i in range(0,5):
    info_returned[i+3] = knet_stats[stat_keys[i]]

  info_returned[8] = mynbm.dataset_generation.calc_haversine(
    p1__=(knet_stats['stla'], knet_stats['stlo']),
    p2__= (knet_stats['evla'], knet_stats['evlo'])
  )
  info_returned[9] = seed_name__
  info_returned[10] = str(stream[0].stats.starttime)
  info_returned[11] = str(stream[0].stats.endtime)
  info_returned[12] = int(info_returned[10].split('-')[0])

  return info_returned