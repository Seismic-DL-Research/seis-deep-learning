import tensorflow as tf

def read_tfr(inFile__, with_year__):
  def decoder(n):
    keys = ['data', 'aavg_ratio', 'overall_ratio', 'magn', 'evla',
            'evlo', 'stla', 'stlo', 'dist',
            'filename', 'start']
            
    dtyp = [tf.float32 for _ in range(9)] + [tf.string for _ in range(2)]
    
    if with_year__:
      keys += ['year']
      dtyp += [tf.int32]

    configs = {
        key: tf.io.FixedLenFeature([], tf.string)
        for key in keys
    }

    parseSingle = tf.io.parse_single_example(n, configs)

    dataStructure = {
        key: tf.io.parse_tensor(parseSingle[key], dtyp[i])
        for i, key in enumerate(keys)
    }
    return (dataStructure)
  return tf.data.TFRecordDataset(inFile__).map(decoder)