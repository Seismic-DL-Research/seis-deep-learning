import numpy as np
import my_notebook_modules as mynbm

def decisor(waveform__, power_windows__, nsta__, nlta__,
            trigger_threshold__, data_threshold__, debug__=False):
  tps, aavg_ratios, scores = [], [], []
  for power_window in power_windows__:
    powerData = mynbm.dataset_generation.power(waveform__, power_window)

    _, tp, _, _, aavg_ratio, score = mynbm.dataset_generation.stalta(
        waveform__,
        powerData,
        nsta__,
        nlta__,
        trigger_threshold__,
        data_threshold__, True, 4)

    tps.append(tp)
    aavg_ratios.append(aavg_ratio)
    scores.append(score)

  tps_ = np.array(tps)
  tps = tps_[tps_ > 0]
  scores = np.array(scores)
  scores = scores[tps_ > 0]
  aavg_ratios = np.array(aavg_ratios)
  aavg_ratios = aavg_ratios[tps_ > 0]

  argsort = np.argsort(scores)

  try:
    if (np.abs(scores[argsort[0]] - scores[argsort[1]]) < 0.8 and
        np.abs(tps[argsort[0]] - tps[argsort[1]]) < 70):
      tps_chosen = 0.7 * tps[argsort[0]] + 0.3 * tps[argsort[1]]
    else:
      tps_chosen = tps[argsort[0]]
    if debug__: return tps, aavg_ratios, scores, int(tps_chosen)
    return int(tps_chosen), aavg_ratios[argsort[0]], scores[argsort[0]]
  except:
    return -1, -1, -1