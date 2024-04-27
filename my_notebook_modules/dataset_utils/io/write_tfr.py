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
  '''
    write_tfr_from_dataset:
      * Used to write tensorflow binary record files from a dataset.

    receive:
      * ds__ <tf.data.Dataset> [0]
        A tf.data.Dataset data to be written.
      * keys__ <list> [?]
        A list of keys and their data type to be written.
        Format of the data in the list: 
          "[YOUR_KEY_NAME].[DATA_TYPE]"
          DATA_TYPE: i32 (tf.int32), f32 (tf.float32), str (tf.string)
          example:
            - ['key_to_destiny.f32', 'key_to_doom.f32', 'key_to_glory.i32']
      * batch_size__ <int> [0]
        The batch size of the writing. Tuning the batch size can speed up the
        writing process.
      * take_size__ <int> [0]
        The size of total batch taken from "ds__" dataset.
      * out_file__ <string> [0]
        The path of the output file 

    return:
      * (0) <0> [0]
        returns nothing.
  '''
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

def write_tfr_from_list(list_tensor__, keys__, out_file__):
  '''
    write_tfr_from_dataset:
      * Used to write tensorflow binary record files from a python list containing
        tensor data.

    receive:
      * list_tensor__ <list> [0]
        A python list that contains tensor data or tensor-like data.
      * keys__ <list> [?]
        A list of keys and their data type to be written.
        Format of the data in the list: 
          "[YOUR_KEY_NAME].[DATA_TYPE]"
          DATA_TYPE: i32 (tf.int32), f32 (tf.float32), str (tf.string)
          example:
            - ['key_to_destiny.f32', 'key_to_doom.f32', 'key_to_glory.i32']
      * out_file__ <string> [0]
        The path of the output file 
        
    return:
      * (0) <0> [0]
        returns nothing.
  '''
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