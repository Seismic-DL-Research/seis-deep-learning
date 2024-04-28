import tensorflow as tf

def disintegrate_complex(input__):
  return input__[:,0], input__[:,1]

def integrate_complex(inputA__, inputB__):
  inputA = tf.expand_dims(inputA__, axis=1)
  inputB = tf.expand_dims(inputB__, axis=1)
  return tf.concat([inputA, inputB], axis=1)