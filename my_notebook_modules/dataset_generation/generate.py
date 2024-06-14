import obspy
import numpy as np
from .calc_stalta import calc_stalta
from .calc_haversine import calc_haversine
from .calc_snr import calc_snr

def generate(list_of_mseeds):
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
    if not tp >= 500: continue
  
    clipped_data = data[tp-100:tp+250]

    magn = knet_stats['mag']
    evla = knet_stats['evla']
    evlo = knet_stats['evlo']
    stla = knet_stats['stla']
    stlo = knet_stats['stlo']
    dist = calc_haversine((evla, evlo), (stla, stlo))
    snr = calc_snr(data[:tp], data[tp:300])

