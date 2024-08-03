'''
WARNING! This model is just for a tutorial purpose!
If you want to actually train your data, you may want to hyper tune or
modify this model. Please act according to your necessity.
'''
import gan_tutorial_modules as gtm
import tensorflow as tf
from gan_tutorial_modules.models import GAN
from tensorflow.keras.layers import Dense, Conv1DTranspose, Input, Flatten, Conv1D, \
AveragePooling1D

class Generative(GAN):
  def __init__(self, gan):
    gan.generative_module = self
    self.gan = gan

    # Generative model
    input_layer = tf.keras.layers.Input(shape=(self.gan.generative_latent_sample_size,))
    expanded = gtm.models.customs.VecAddChannel()(input_layer)

    # Load custom Leaky ReLU
    ab_leaky_relu = gtm.models.customs.ab_leaky_relu

    # begin deconvolution 
    cont = Conv1DTranspose(filters=2, kernel_size=7, 
                          activation=ab_leaky_relu(0.1,1))(expanded)
    cont = Conv1DTranspose(filters=4, kernel_size=12,
                          activation=ab_leaky_relu(0.1,1))(cont)
    cont = Conv1DTranspose(filters=8, kernel_size=16,
                          activation=ab_leaky_relu(0.1,1))(cont)
    cont = Conv1DTranspose(filters=6, kernel_size=12,
                          activation=ab_leaky_relu(0.1,1))(cont)

    # exit deconvolution, begin fully-connected
    cont = Flatten()(cont)
    cont = Dense(400, activation=ab_leaky_relu(0.1,1))(cont)
    cont = Dense(self.gan.window_length, activation='tanh')(cont)

    # construct the model
    self.model = tf.keras.Model(inputs=[input_layer], outputs=[cont])


  @tf.function # adding decorator to accelerate training
  def loss(self, d_model):
    # d_model is the discriminative model

    # Define epsilon to avoid nan
    eps = 1e-6

    # We generate the artificial P wave.
    X0 = tf.random.normal(shape=self.gan.latent_shape, 
                          mean=self.gan.generative_latent_sample_mean, 
                          stddev=self.gan.generative_latent_sample_stdev)
    XG = self.model(X0)

    # Total loss
    loss = tf.math.log(d_model(XG) + eps) # B x 1
    loss = tf.math.reduce_sum(loss, axis=0) # shape: 1
    loss = -1 * loss[0]
    
    return loss

  @tf.function # adding decorator to accelerate training
  def update_trainable_tensors(self, d_model):
    # d_model is the discriminative model
    
    with tf.GradientTape() as g:
      # Calculating L_g
      g_loss = self.loss(d_model)

      # Calculating ∂L_g/∂θ_g
      grad = g.gradient(g_loss, self.model.trainable_variables)

      # Updating θ_g := θ_g - α(∂L_g/∂θ_g)
      self.gan.generative_optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
    
    return g_loss

  def update_model(self, new_model):
    contains_error = False

    # check if model's I/O shape is valid
    input_shape = new_model.layers[0].output.shape[1]
    output_shape = new_model.layers[-1].output.shape[1]

    try:
      assert input_shape == self.gan.generative_latent_sample_size
    except:
      print(f'Invalid input shape! Expected {self.gan.generative_latent_sample_size} but obtained {input_shape}')
      contains_error = True
    
    try:
      assert output_shape == self.gan.window_length
    except:
      print(f'Invalid output shape! Expected {self.gan.window_length} but obtained {output_shape}')
      contains_error = True

    # update model if valid
    if not contains_error: self.model = new_model