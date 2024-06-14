import obspy
import numpy as np
from .calc_stalta import calc_stalta
from .calc_haversine import calc_haversine
from .calc_snr import calc_snr

def generate(list_of_mseeds):
  # data | dist | evla | evlo | stla | stlo | snr | magn
  generated_data = [[], [], [], [], [], [], [], []]

  for mseed in list_of_mseeds:
    # KiK-net miniSEED: 1 trace per mseed
    stream = obspy.read(mseed)[0]
    knet_stats = stream.stats.knet
    data = stream.data
    _, tp = calc_stalta(data, 20, 300, 5)

    # no proper P phase discerned
    if np.shape(tp)[0] == 0: continue

    # need at least 5 second noise
    tp = tp[0]
    if not tp >= 400: continue
  
    generated_data[0].append(data[tp-100:tp+250])
    generated_data[2].append(knet_stats['evla'])
    generated_data[3].append(knet_stats['evlo'])
    generated_data[4].append(knet_stats['stla'])
    generated_data[5].append(knet_stats['stlo'])
    generated_data[1].append(calc_haversine((generated_data[2][-1], generated_data[3][-1]), 
                                            (generated_data[4][-1], generated_data[5][-1])))
    generated_data[6].append(calc_snr(data[:tp], data[tp:300]))
    generated_data[7].append(knet_stats['mag'])
  
  return generated_data

