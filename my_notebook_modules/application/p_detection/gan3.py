import tensorflow as tf

@tf.keras.utils.register_keras_serializable(
    package="thesis-gan",
    name="SplitLayer"
)
class SplitLayer(tf.keras.Layer):
  def __init__(sf):
    super(SplitLayer, sf).__init__()

  def build(sf, input_shape__):
    pass

  def call(sf, inputs__):
    # inputs__: N x H x C
    split_tensor = tf.split(inputs__, num_or_size_splits=3, axis=2)
    return split_tensor
  
class GAN3():
  def __init__(sf):
    model_loc = 'seis-deep-learning/my_notebook_modules/application/p_detection/keras/&.keras'
    gModel = sf.INTERNAL_G()
    dModel = sf.INTERNAL_D()
    
    gModel.load_weights(model_loc.replace('&', 'gan3_g'))
    dModel.load_weights(model_loc.replace('&', 'gan3_d'))

    sf.g_model = gModel
    sf.d_model = dModel
  
  def INTERNAL_G(sf):
    tf.random.set_seed(6969)
    def functional_model():
      input = tf.keras.Input(shape=(50,1))
      x = tf.keras.layers.BatchNormalization()(input)
      x = tf.keras.layers.Conv1DTranspose(filters=11, kernel_size=32, name='conv1d_transpose1')(x)
      x = tf.keras.layers.AveragePooling1D(pool_size=2, name='avgpool1d1')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Conv1DTranspose(filters=5, kernel_size=64, name='conv1d_transpose2')(x)
      x = tf.keras.layers.AveragePooling1D(pool_size=2, name='avgpool1d2')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Flatten(name='flatten1')(x)
      x = tf.keras.layers.Dense(300, activation='leaky_relu', name='dense1')(x)
      x = tf.keras.layers.Dense(400, activation='leaky_relu', name='dense2')(x)
      x = tf.keras.layers.Dense(350, activation='tanh', name='dense3')(x)
      x = tf.keras.layers.Reshape((350, 1))(x)
      return tf.keras.Model(inputs=input, outputs=x)

    func_model = functional_model()

    init_inputs = tf.keras.Input(shape=(50,3))
    inputs = SplitLayer()(init_inputs)

    ud_output = func_model(inputs[0])
    ew_output = func_model(inputs[1])
    ns_output = func_model(inputs[2])

    out = tf.keras.layers.Concatenate(axis=-1)([ud_output, ew_output, ns_output])

    return tf.keras.Model(inputs=[init_inputs], outputs=[out], name='generator_model')

  def INTERNAL_D(sf):
    tf.random.set_seed(6969)
    def functional_model():
      input = tf.keras.Input(shape=(350,1))
      x = tf.keras.layers.BatchNormalization()(input)
      x = tf.keras.layers.Conv1D(filters=10, kernel_size=10, name='conv1d1')(x)
      x = tf.keras.layers.AveragePooling1D(pool_size=2, name='avgpool1d1')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Conv1D(filters=3, kernel_size=40, name='conv1d2')(x)
      x = tf.keras.layers.Flatten(name='flatten1')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Dense(300, activation='leaky_relu', name='dense1')(x)
      x = tf.keras.layers.Dense(200, activation='leaky_relu', name='dense2')(x)
      x = tf.keras.layers.Dense(50, activation='leaky_relu', name='dense3')(x)
      x = tf.keras.layers.Dense(5, activation='sigmoid', name='dense4')(x)
      return tf.keras.Model(inputs=input, outputs=x)


    func_model = functional_model()

    init_inputs = tf.keras.Input(shape=(350,3))
    inputs = SplitLayer()(init_inputs)

    ud_output = func_model(inputs[0])
    ew_output = func_model(inputs[1])
    ns_output = func_model(inputs[2])

    out = tf.keras.layers.Concatenate(axis=-1)([ud_output, ew_output, ns_output])
    out = tf.keras.layers.Dense(1, activation='sigmoid')(out)

    return tf.keras.Model(inputs=[init_inputs], outputs=[out], name='discriminator_model')
  
  @tf.function
  def predict_single(sf, data_z, data_e, data_n):
    data_z = tf.expand_dims(data_z, axis=1)
    data_z = data_z/tf.math.reduce_max(data_z)
    data_e = tf.expand_dims(data_e, axis=1)
    data_e = data_e/tf.math.reduce_max(data_e)
    data_n = tf.expand_dims(data_n, axis=1)
    data_n = data_n/tf.math.reduce_max(data_n)
    zen = tf.concat([data_z, data_e, data_n], axis=1)
    zen = tf.expand_dims(zen, axis=0)

    prediction = sf.d_model(zen)[0,0]
    return prediction

  @tf.function
  def predict_sliding(sf, data_z, data_e, data_n, freq, start_sample, end_sample):
    step = int(100/freq)
    step_indices = start_sample
    predictions = []
    while step_indices + step <= end_sample - 350:
      predictions.append(sf.predict_single(data_z[step_indices:step_indices+350], 
                                           data_e[step_indices:step_indices+350], 
                                           data_n[step_indices:step_indices+350]))
      step_indices += step
    
    return predictions

  def predict_batch(sf):
    pass