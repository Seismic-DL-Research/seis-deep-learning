import tensorflow as tf
from tqdm import tqdm

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
                           take_size__, out_file__, batches_per_file__=-1):
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
      * batches_per_file__ <int> [0] {optional}
        Used for sharding. The value tells how many batches are recorded per file.
        Total data elements (singular data) is batches_per_file__ * batch_size__
        DEFAULT = -1 (take all batches)

    return:
      * (0) <0> [0]
        returns nothing.
  '''
  keys, keys_type = get_keys_and_types(keys__)
  init_filename = (out_file__ if batches_per_file__ == -1 
                  else f'{".".join(out_file__.split(".")[:-1])}:0.tfr')
  bar = tqdm(total=1, position=0, bar_format='[{elapsed}] {desc}')
  f = tf.io.TFRecordWriter(init_filename)
  batch_rounds = 0

  bar.set_description_str(f'Initializing')
  for dsElement in ds__.batch(batch_size__).take(take_size__):
    # close the current file and open the new file if batches_per_file__ is
    # enabled (!=1).
    if batches_per_file__ != -1 and batch_rounds % batches_per_file__ == 0:
      f.close()
      f = tf.io.TFRecordWriter(
        f'{".".join(out_file__.split(".")[:-1])}:{int(batch_rounds/batches_per_file__)}.tfr')
    
    collective = tf.train.Features(
        feature = {
            key: featurize(serialize(dsElement[key], keys_type[i]))
            for i, key in enumerate(keys)
        }
    )
    # write serialized tensor (in bytes).
    record_bytes = tf.train.Example(features=collective).SerializeToString()
    f.write(record_bytes)

    # increase the batch_rounds number.
    batch_rounds += 1

    # update tqdm
    bar.set_description_str(f'Batch Rounds: {batch_rounds}')

    

  bar.close()

  f.close()

def write_tfr_from_list(list_tensor__, keys__, out_file__):
  '''
    write_tfr_from_list:
      * Used to write tensorflow binary record files from a python list that is
        tensor-able (able to converted into tensor).

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