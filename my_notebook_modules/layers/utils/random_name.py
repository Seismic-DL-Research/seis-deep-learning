import tensorflow as tf

@tf.function
def random_name(layer_type__):
  random_number = tf.strings.as_string(tf.random.uniform((1,), 100000,999999, tf.int32))
  layer_type = tf.expand_dims(layer_type__, axis=0)
  return tf.strings.join([layer_type, ['.'], random_number])[0]
