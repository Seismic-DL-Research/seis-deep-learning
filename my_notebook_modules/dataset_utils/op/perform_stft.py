import tensorflow as tf
import tensorflow.math as tfm
import my_notebook_modules as mynbm
from scipy.signal import stft

def perform_stft(dataset__, take_size__, batch_size__,
                 nperseg__, noverlap__, save_stft__):
  accumulated_data = [[], [], []]
  data_keys = ['data', 'dist', 'magn']
  data_dtyp = [tf.float32 for _ in range(3)]
  write_incidents = 0

  for i in dataset__.take(take_size__):
    f, t, Z = stft(i['data'], fs=100, nperseg=nperseg__, noverlap=noverlap__)
    ZR = tf.expand_dims(tfm.real(Z), axis=1)
    ZJ = tf.expand_dims(tfm.imag(Z), axis=1)
    Z = tf.concat([ZR, ZJ], axis=1)

    accumulated_data[0].append(Z)
    accumulated_data[1].append(i['dist'])
    accumulated_data[2].append(i['magn'])

    if (len(accumulated_data[0]) >= batch_size__):
      write_incidents += 1
      mynbm.dataset_utils.io.write_tfr_from_list(accumulated_data, 
                                                 data_keys,
                                                 data_dtyp,
                                                 f'{save_stft__}-{write_incidents}.tfr')
      accumulated_data = [[], [], []]

  if (len(accumulated_data[0]) != 0):
    write_incidents += 1
    mynbm.dataset_utils.io.write_tfr_from_list(accumulated_data, 
                                                data_keys,
                                                data_dtyp,
                                                f'{save_stft__}-{write_incidents}.tfr')
