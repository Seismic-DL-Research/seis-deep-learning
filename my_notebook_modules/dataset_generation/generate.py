import obspy
import numpy as np
import my_notebook_modules as mynbm
from .calc_stalta import calc_stalta
from .calc_haversine import calc_haversine
from .calc_snr import calc_snr

def generate(list_of_mseeds, tfr_dest, elem_per_tfr):
  # 0      1      2      3      4      5      6     7      8      9      10 | 11
  # data | data | dist | evla | evlo | stla | stlo | snr | magn | stnm | time | year
  generated_data = [[], [], [], [], [], [], [], [], [], [], [], []]
  noises = []
  keys = ['data.f32', 'dist.f32', 'evla.f32', 'evlo.f32', 
          'stla.f32', 'stlo.f32', 'snr.f32', 'magn.f32',
          'stnm.str', 'time.str',  'year.i32']
  counts = 0

  for mseed in list_of_mseeds:
    # KiK-net miniSEED: 1 trace per mseed
    stream = obspy.read(mseed).detrend(type='constant')[0]
    knet_stats = stream.stats.knet
    data = stream.data * stream.stats.calib * 100
    _, tp = calc_stalta(data, 30, 200, 5)

    # no proper P phase discerned
    if np.shape(tp)[0] == 0: continue

    # need at least 5 second noise
    tp = tp[0]
    clipped_data = data[tp-100:tp+250]
    if not tp >= 400 or len(clipped_data) != 350: continue
    clipped_noise = data[0:350]
    noises.append(clipped_noise)
    generated_data[0].append(clipped_data)
    generated_data[2].append(knet_stats['evla'])
    generated_data[3].append(knet_stats['evlo'])
    generated_data[4].append(knet_stats['stla'])
    generated_data[5].append(knet_stats['stlo'])
    generated_data[1].append(calc_haversine((generated_data[2][-1], generated_data[3][-1]), 
                                            (generated_data[4][-1], generated_data[5][-1])))
    generated_data[6].append(calc_snr(data[:tp], data[tp:tp+300]))
    generated_data[7].append(knet_stats['mag'])

    name = mseed.split('/')[-1].split('.')[0]
    generated_data[8].append(name)
    generated_data[9].append(str(stream.stats.starttime))
    generated_data[10].append(int(str(stream.stats.starttime).split('-')[0]))  
    counts += 1

    # if exceeds elem_per_tfr, write and reset memory
    if (counts % elem_per_tfr == 0):
      tfr_name_P = f'{tfr_dest}:{int(counts/elem_per_tfr)}-P.tfr'
      tfr_name_N = f'{tfr_dest}:{int(counts/elem_per_tfr)}-N.tfr'
      mynbm.dataset_utils.io.write_tfr_from_list(generated_data, keys, tfr_name_P)
      generated_data[0] = noises
      mynbm.dataset_utils.io.write_tfr_from_list(generated_data, keys, tfr_name_N)
      generated_data = [[], [], [], [], [], [], [], [], [], [], []]
      noises = []

  if generated_data[0] != []:
      tfr_name_P = f'{tfr_dest}:{int(counts/elem_per_tfr) + 1}-P.tfr'
      tfr_name_N = f'{tfr_dest}:{int(counts/elem_per_tfr) + 1}-N.tfr'
      mynbm.dataset_utils.io.write_tfr_from_list(generated_data, keys, tfr_name_P)
      generated_data[0] = noises
      mynbm.dataset_utils.io.write_tfr_from_list(generated_data, keys, tfr_name_N)
      generated_data = [[], [], [], [], [], [], [], [], [], [], []]

  return 1

