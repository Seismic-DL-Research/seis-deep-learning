import tensorflow as tf

def define_type(x):
  if (x == 'f32'): return tf.float32
  if (x == 'str'): return tf.string
  if (x == 'i32'): return tf.int32
  return -1

def get_keys_and_types(keys__):
  keys = [key.split('.')[0] for key in keys__]
  keys_type = [define_type(key.split('.')[1]) for key in keys__]

@tf.autograph.experimental.do_not_convert
def read_tfr(inFile__, keys__, dtyp__):
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