import tensorflow as tf

@tf.keras.utils.register_keras_serializable(
    package="thesis-gan",
    name="ExtendDimension"
)
class INTERNAL_ExtendDimension(tf.keras.Layer):
  def __init__(sf):
    super().__init__()

  def build(sf, input_shape__):
    pass

  def call(sf, inputs__):
    return tf.expand_dims(inputs__, axis=2) # N x D x 1

def INTERNAL_G():
  proposedLayers = [
      tf.keras.Input(shape=(50,)),
      INTERNAL_ExtendDimension(),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv1DTranspose(filters=11,
                                      kernel_size=32,
                                      name='conv1d_transpose1'),
      tf.keras.layers.AveragePooling1D(pool_size=2,
                                       name='avgpool1d1'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv1DTranspose(filters=5,
                                      kernel_size=64,
                                      name='conv1d_transpose2'),
      tf.keras.layers.AveragePooling1D(pool_size=2,
                                       name='avgpool1d2'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Flatten(name='flatten1'),
      tf.keras.layers.Dense(300, activation='leaky_relu', name='dense1'),
      tf.keras.layers.Dense(400, activation='leaky_relu', name='dense2'),
      tf.keras.layers.Dense(350, activation='tanh', name='dense3'),
      tf.keras.layers.BatchNormalization()
  ]

  firstLayer = proposedLayers[0]
  for i, proposedLayer in enumerate(proposedLayers[1:]):
    if i == 0: connectedLayer = proposedLayer(firstLayer)
    else: connectedLayer = proposedLayer(prevConnectedLayer)
    prevConnectedLayer = connectedLayer

  return tf.keras.Model(inputs=[firstLayer], outputs=[connectedLayer],
                        name='generator_model')

def INTERNAL_D():
  proposedLayers = [
      tf.keras.Input(shape=(350,)),
      INTERNAL_ExtendDimension(),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv1D(filters=10,
                            kernel_size=10,
                            name='conv1d1'),
      tf.keras.layers.AveragePooling1D(pool_size=2,
                                        name='avgpool1d1'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv1D(filters=3,
                              kernel_size=40,
                              name='conv1d2'),
      tf.keras.layers.Flatten(name='flatten1'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(300, activation='leaky_relu', name='dense1'),
      tf.keras.layers.Dense(200, activation='leaky_relu', name='dense2'),
      tf.keras.layers.Dense(50, activation='leaky_relu', name='dense3'),
      tf.keras.layers.Dense(1, activation='sigmoid', name='dense4')
  ]

  firstLayer = proposedLayers[0]
  for i, proposedLayer in enumerate(proposedLayers[1:]):
    if i == 0: connectedLayer = proposedLayer(firstLayer)
    else: connectedLayer = proposedLayer(prevConnectedLayer)
    prevConnectedLayer = connectedLayer

  return tf.keras.Model(inputs=[firstLayer], outputs=[connectedLayer],
                        name='discriminator_model')


def gan1():
  gModel = INTERNAL_G()
  dModel = INTERNAL_D()
  gModel.load_weights('./keras/gan1_g.keras')
  dModel.load_weights('./keras/gan1_d.keras')
  return gModel, dModel