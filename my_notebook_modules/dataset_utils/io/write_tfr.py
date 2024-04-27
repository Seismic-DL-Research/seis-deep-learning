import tensorflow as tf

def serialize(data__, dtype__):
  return tf.io.serialize_tensor(tf.convert_to_tensor(data__, dtype__))

def featurize(value__):
  return tf.train.Feature(
      bytes_list = tf.train.BytesList(value=[value__.numpy()])
  )

def define_type(x):
  if (x == 'f32'): return tf.float32
  if (x == 'str'): return tf.string
  if (x == 'i32'): return tf.int32
  return -1

def get_keys_and_types(keys__):
  keys = [key.split('.')[0] for key in keys__]
  keys_type = [define_type(key.split('.')[1]) for key in keys__]
  return keys, keys_type

def write_tfr_from_dataset(ds__, keys__, batch_size__, 
                           take_size__, out_file__):
  keys, keys_type = get_keys_and_types(keys__)

  with tf.io.TFRecordWriter(f'{out_file__}') as f:
    for dsElement in ds__.batch(batch_size__).take(take_size__):
      collective = tf.train.Features(
          feature = {
              key: featurize(serialize(dsElement[key], keys_type[i]))
              for i, key in enumerate(keys)
          }
      )
      record_bytes = tf.train.Example(features=collective).SerializeToString()

      f.write(record_bytes)

def write_tfr_from_list(list_tensor__, keys__, dtyp__, out_file__):
  keys, keys_type = get_keys_and_types(keys__)

  collective = tf.train.Features(
    feature={
      key: featurize(serialize(list_tensor__[i], keys_type[i]))
      for i, key in enumerate(keys)
    }
  )
  record_bytes = tf.train.Example(features=collective).SerializeToString()

  with tf.io.TFRecordWriter(f'{out_file__}') as f:
    f.write(record_bytes)