import tensorflow as tf

''' (personal note)
    keys = ['data', 'aavg_ratio', 'overall_ratio', 'magn', 'evla',
            'evlo', 'stla', 'stlo', 'dist',
            'filename', 'start']
            
    dtyp = [tf.float32 for _ in range(9)] + [tf.string for _ in range(2)]
'''
@tf.autograph.experimental.do_not_convert
def read_tfr(inFile__, keys__, dtyp__):
  def decoder(n):
    configs = {
        key: tf.io.FixedLenFeature([], tf.string)
        for key in keys__
    }

    parseSingle = tf.io.parse_single_example(n, configs)

    dataStructure = {
        key: tf.io.parse_tensor(parseSingle[key], dtyp__[i])
        for i, key in enumerate(keys__)
    }
    return (dataStructure)
  return tf.data.TFRecordDataset(inFile__).map(decoder)