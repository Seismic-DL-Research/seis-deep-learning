import tensorflow as tf

def serialize(data__, dtype__):
  return tf.io.serialize_tensor(tf.convert_to_tensor(data__, dtype__))

def featurize(value__):
  return tf.train.Feature(
      bytes_list = tf.train.BytesList(value=[value__.numpy()])
  )

def write_tfr(ds__, outFile__):
  keys = ['data', 'aavg_ratio', 'overall_ratio', 'magn', 'evla',
          'evlo', 'stla', 'stlo', 'dist',
          'filename', 'start', 'year']

  dtyp = [tf.float32 for _ in range(9)] + [tf.string for _ in range(2)] + [tf.int32]

  with tf.io.TFRecordWriter(f'{outFile__}') as f:
    serialized = {}

    for dsElement in ds__.take(-1):
      collective = tf.train.Features(
          feature = {
              keys[i]: featurize(serialize(dsElement[dsKey], dtyp[i]))
              for i, dsKey in enumerate(dsElement)
          }
      )
      record_bytes = tf.train.Example(features=collective).SerializeToString()

      f.write(record_bytes)