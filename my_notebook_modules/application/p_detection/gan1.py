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

class GAN1():
  def __init__(sf):
    model_loc = 'seis-deep-learning/my_notebook_modules/application/p_detection/keras/&.keras'
    gModel = sf.INTERNAL_G()
    dModel = sf.INTERNAL_D()
    
    gModel.load_weights(model_loc.replace('&', 'gan1_g'))
    dModel.load_weights(model_loc.replace('&', 'gan1_d'))
    sf.g_model = gModel 
    sf.d_model = dModel

  def INTERNAL_G(sf):
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

  def INTERNAL_D(sf):
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

  def predict_single(sf, data):
    data = tf.expand_dims(data, axis=0)
    prediction = sf.d_model(data)[0,0]
    return float(prediction)

  def predict_sliding(sf, data, freq, start_sample, end_sample):
    data = tf.expand_dims(data, axis=0)
    print(data.shape)
    step = int(100/freq)
    step_indices = start_sample
    predictions = []
    while step_indices + step <= end_sample - 350:
      step_indices += step
      predictions.append(sf.predict_single(data[step_indices:step_indices+350]))
    
    return predictions

  def predict_batch(sf):
    pass