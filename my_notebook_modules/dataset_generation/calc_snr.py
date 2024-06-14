import numpy as np

def calc_snr(noise_data, signal_data):
  rms_noisex = np.sqrt(np.mean(noise_data**2))
  rms_signal = np.sqrt(np.mean(signal_data**2))
  return 20 * np.log(rms_signal / rms_noise) / np.log(10)