import tensorflow as tf 
import my_notebook_modules as mynbm
from scipy.signal import stft

@tf.keras.utils.register_keras_serializable(
    package="thesis-gan",
    name="ToComplex"
)
class ToComplex(tf.keras.Layer):
  def __init__(sf):
    super(ToComplex, sf).__init__()

  def build(sf, input_shape__):
    pass

  def call(sf, input):
    output = tf.expand_dims(tf.transpose(input, perm=[0,3,1,2]), axis=-1)
    return output

class COMPLEXGAN():

  def __init__(sf):
    model_loc = 'seis-deep-learning/my_notebook_modules/application/p_detection/keras/&.keras'
    gModel = sf.INTERNAL_G()
    dModel = sf.INTERNAL_D()
    
    # gModel.load_weights(model_loc.replace('&', 'complexgan_g'))
    dModel.load_weights(model_loc.replace('&', 'complexgan_d'))
    sf.d_model = dModel
  
  def INTERNAL_G(sf):
    input = tf.keras.Input(shape=(20,24,1))
    ly = tf.keras.layers.Conv2DTranspose(6, (3,6), activation='leaky_relu')(input)
    ly = tf.keras.layers.Conv2DTranspose(6, (5,7), activation='leaky_relu')(ly)
    ly = tf.keras.layers.Conv2DTranspose(8, (5,7), activation='leaky_relu')(ly)
    ly = tf.keras.layers.Conv2DTranspose(8, (5,7), activation='leaky_relu')(ly)
    ly = tf.keras.layers.AveragePooling2D((2,2))(ly)
    ly = tf.keras.layers.BatchNormalization()(ly)

    ly0 = tf.keras.layers.Conv2DTranspose(12, (9,12), activation='leaky_relu', padding='SAME')(ly)
    ly = tf.keras.layers.Conv2DTranspose(12, (9,12), activation='leaky_relu', padding='SAME')(ly0)
    ly = tf.keras.layers.Conv2DTranspose(12, (9,12), activation='leaky_relu', padding='SAME')(ly)
    ly = tf.keras.layers.Add()([ly0, ly])

    ly = tf.keras.layers.UpSampling2D((2,2))(ly)
    ly = tf.keras.layers.AveragePooling2D((2,2))(ly)
    ly = tf.keras.layers.BatchNormalization()(ly)

    ly0 = tf.keras.layers.Conv2DTranspose(14, (9,12), activation='leaky_relu', padding='SAME')(ly)
    ly = tf.keras.layers.Conv2DTranspose(14, (9,12), activation='leaky_relu', padding='SAME')(ly0)
    ly = tf.keras.layers.Conv2DTranspose(14, (9,12), activation='leaky_relu', padding='SAME')(ly)
    ly = tf.keras.layers.Add()([ly0, ly])

    ly = tf.keras.layers.UpSampling2D((2,2))(ly)
    ly = tf.keras.layers.BatchNormalization()(ly)

    ly0 = tf.keras.layers.Conv2DTranspose(16, (6,8), activation='leaky_relu')(ly)
    ly = tf.keras.layers.Conv2DTranspose(16, (6,6), activation='leaky_relu')(ly0)
    ly = tf.keras.layers.BatchNormalization()(ly)

    ly = tf.keras.layers.Conv2DTranspose(12, (9,12), activation='leaky_relu', padding='SAME')(ly)
    ly = tf.keras.layers.Conv2DTranspose(6, (12,12), activation='leaky_relu', padding='SAME')(ly)
    ly = tf.keras.layers.BatchNormalization()(ly)

    ly = tf.keras.layers.Conv2DTranspose(6, (12,12), activation='leaky_relu', padding='SAME')(ly)
    ly = tf.keras.layers.Conv2DTranspose(2, (8,3), activation='leaky_relu')(ly)
    ly = tf.keras.layers.BatchNormalization()(ly)

    ly = ToComplex()(ly)
    mdl = tf.keras.Model(inputs=[input], outputs=[ly])
    return mdl

  def INTERNAL_D(sf):
    tf.random.set_seed(6969)
    leaky_relu = mynbm.layers.utils.complex_leaky_relu(0.1)

    input = tf.keras.Input(shape=(2,51,60,1))
    ly = mynbm.layers.complex_conv_2d((3,3), 8, 'VALID', leaky_relu)(input)
    ly = mynbm.layers.complex_conv_2d((3,3), 8, 'VALID', leaky_relu)(ly)
    ly = mynbm.layers.complex_avg_pool_2d((2,2), 'VALID', leaky_relu)(ly)
    ly = tf.keras.layers.BatchNormalization()(ly)

    ly0 = mynbm.layers.complex_conv_2d((3,3), 14, 'SAME', leaky_relu)(ly)
    ly = mynbm.layers.complex_conv_2d((3,3), 14, 'SAME', leaky_relu)(ly0)
    ly = mynbm.layers.complex_conv_2d((3,3), 14, 'SAME', leaky_relu)(ly)
    ly = tf.keras.layers.Add()([ly0,ly])

    ly0 = mynbm.layers.complex_conv_2d((3,3), 4, 'SAME', leaky_relu)(ly)
    ly = mynbm.layers.complex_conv_2d((3,3), 4, 'SAME', leaky_relu)(ly0)
    ly = mynbm.layers.complex_conv_2d((3,3), 4, 'SAME', leaky_relu)(ly)
    ly = tf.keras.layers.Add()([ly0,ly])
    ly = mynbm.layers.complex_avg_pool_2d((2,2), 'VALID', leaky_relu)(ly)

    ly0 = mynbm.layers.complex_conv_2d((3,3), 4, 'SAME', leaky_relu)(ly)
    ly = mynbm.layers.complex_conv_2d((3,3), 4, 'SAME', leaky_relu)(ly0)
    ly = mynbm.layers.complex_conv_2d((3,3), 4, 'SAME', leaky_relu)(ly)
    ly = tf.keras.layers.Add()([ly0,ly])
    ly = mynbm.layers.complex_avg_pool_2d((2,2), 'VALID', leaky_relu)(ly)

    ly = tf.keras.layers.BatchNormalization()(ly)
    ly = mynbm.layers.complex_flatten()(ly)
    ly = mynbm.layers.complex_conjugate()(ly)
    ly = tf.keras.layers.Dense(256, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.L2(0.1))(ly)
    ly = tf.keras.layers.Dense(128, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.L2(0.1))(ly)
    ly = tf.keras.layers.Dense(10, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.L2(0.1))(ly)
    ly = tf.keras.layers.Dense(1, activation='sigmoid')(ly)


    mdl = tf.keras.Model(inputs=[input], outputs=[ly])
    return mdl
  
  def predict_single(sf, data):
    # convert into STFT domain
    data = data / tf.math.reduce_max(data)
    _, _, Zxx = stft(data, fs=100, nperseg=100, noverlap=94)
    Zxx_real = tf.expand_dims(tf.math.real(Zxx), axis=0)
    Zxx_imag = tf.expand_dims(tf.math.real(Zxx), axis=0)
    Zxx_comp = tf.expand_dims(tf.concat([Zxx_real, Zxx_imag], axis=0), axis=-1)
    Zxx = tf.expand_dims(Zxx_comp, axis=0)
    prediction = sf.d_model(Zxx)[0,0]
    return float(prediction)

  def predict_sliding(sf, data, freq, start_sample, end_sample):
    step = int(100/freq)
    step_indices = start_sample
    predictions = []
    while step_indices + step <= end_sample - 350:
      predictions.append(sf.predict_single(data[step_indices:step_indices+350]))
      step_indices += step
    
    return predictions

  def predict_batch(sf):
    pass