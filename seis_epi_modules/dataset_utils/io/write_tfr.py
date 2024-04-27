import tensorflow as tf

def serialize(data__, dtype__):
  return tf.io.serialize_tensor(tf.convert_to_tensor(data__, dtype__))

def featurize(value__):
  return tf.train.Feature(
      bytes_list = tf.train.BytesList(value=[value__.numpy()])
  )

# keys = ['data', 'aavg_ratio', 'overall_ratio', 'magn', 'evla',
#           'evlo', 'stla', 'stlo', 'dist',
#           'filename', 'start', 'year']

# dtyp = [tf.float32 for _ in range(9)] + [tf.string for _ in range(2)] + [tf.int32]
def write_tfr_from_dataset(ds__, keys__, dtyp__, out_file__):
  with tf.io.TFRecordWriter(f'{out_file__}') as f:

    for dsElement in ds__.take(-1):
      collective = tf.train.Features(
          feature = {
              keys__[i]: featurize(serialize(dsElement[dsKey], dtyp__[i]))
              for i, dsKey in enumerate(dsElement)
          }
      )
      record_bytes = tf.train.Example(features=collective).SerializeToString()

      f.write(record_bytes)

def write_tfr_from_listTensor(list_tensor__, keys__, dtyp__, out_file__):
  collective = tf.train.Features(
    feature={
      tkey: featurize(serialize(list_tensor__[i], dtyp__[i]))
      for i, tkey in enumerate(keys__)
    }
  )
  record_bytes = tf.train.Example(features=collective).SerializeToString()

  with tf.io.TFRecordWriter(f'{out_file__}') as f:
    f.write(record_bytes)