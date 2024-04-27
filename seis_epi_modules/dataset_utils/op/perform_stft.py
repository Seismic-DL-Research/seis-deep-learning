import tensorflow as tf
import tensorflow.math as tfm
import seis_epi_modules as sem
from scipy.signal import stft

def perform_stft(dataset__, take_size__, batch_size__):
  accumulated_data = [[], [], []]
  data_keys = ['data', 'dist', 'magn']
  data_dtyp = [tf.float32 for _ in range(3)]

  for i in dataset__.take(take_size__):
    f, t, Z = stft(i['data'], nperseg=122, noverlap=105)
    ZR = tf.expand_dims(tfm.real(Z), axis=1)
    ZJ = tf.expand_dims(tfm.imag(Z), axis=1)
    Z = tf.concat([ZR, ZJ], axis=1)

    accumulated_data[0].append(Z)
    accumulated_data[1].append(i['dist'])
    accumulated_data[2].append(i['magn'])

    if (len(accumulated_data[0]) > batch_size__):
      sem.dataset_utils.io.write_tfr_from_list(accumulated_data, 
                                               data_keys,
                                               data_dtyp,
                                               '../stft_debug.tfr')
