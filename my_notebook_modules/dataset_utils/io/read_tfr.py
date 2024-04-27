import tensorflow as tf

def define_type(x):
  if (x == 'f32'): return tf.float32
  if (x == 'str'): return tf.string
  if (x == 'i32'): return tf.int32
  return -1

def get_keys_and_types(keys__):
  keys = [key.split('.')[0] for key in keys__]
  keys_type = [define_type(key.split('.')[1]) for key in keys__]
  return keys, keys_type

@tf.autograph.experimental.do_not_convert
def read_tfr(inFile__, keys__):
  '''
    read_tfr:
      * used to read tensorflow binary record files.

    receive:
      * in_File__ <string/list> [0]
        a string/list containing the path to the tensorflow binary record file(s).
      * keys__ <list> [?]
        a list of keys and their data type that is present in the binary file(s).
        Format of the data in the list: 
          "[YOUR_KEY_NAME].[DATA_TYPE]"
          DATA_TYPE: i32 (tf.int32), f32 (tf.float32), str (tf.string)
          example:
            - ['key_to_destiny.f32', 'key_to_doom.f32', 'key_to_glory.i32']

    return:
      * (a dataset) <tf.data.Dataset> [0]
        a dataset that is ready to be read by tf.data.Dataset module.
  '''
  keys, keys_type = get_keys_and_types(keys__)

  def decoder(n):
    configs = {
        key: tf.io.FixedLenFeature([], tf.string)
        for key in keys
    }

    parseSingle = tf.io.parse_single_example(n, configs)

    dataStructure = {
        key: tf.io.parse_tensor(parseSingle[key], keys_type[i])
        for i, key in enumerate(keys)
    }
    return (dataStructure)
  return tf.data.TFRecordDataset(inFile__).map(decoder)