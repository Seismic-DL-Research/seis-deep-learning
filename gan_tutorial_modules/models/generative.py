'''
WARNING! This model is just for a tutorial purpose!
If you want to actually train your data, you may want to hyper tune or
modify this model. Please act according to your necessity.
'''
import gan_tutorial_modules as gtm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1DTranspose, Input, Flatten, Conv1D, \
AveragePooling1D

class Generative(gtm.models.GAN):
  def generative(self, gan):
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
  def loss(self, X_G):
    # Variable X_G is R^(BxN) with B is the batch size.

    # Total loss
    loss = tf.math.log(sf.model(X_G)) # B x 1
    loss = tf.math.reduce_sum(loss, axis=0) # shape: 1
    loss = -1 * loss[0]
    
    return loss

  @tf.function # adding decorator to accelerate training
  def update_trainable_tensors(self, d_model):
    # d_model is the discriminative model

    # We generate the artificial P wave.
    # We have defined the value of B, L, mu, and sigma earlier.
    X0 = tf.random.normal(shape=(self.gan.batch_size, self.gan.window_length), 
                          mean=self.gan.generative_latent_sample_mean, 
                          stddev=self.gan.generative_latent_sample_stdev)
    
    with tf.GradientTape() as g:
      # The generation of the artificial waveform should be put here
      # so the gradient can be calculated.
      X_G = self.model(X0)

      # Calculating L_g
      g_loss = self.loss(X_G, d_model)

      # Calculating ∂L_g/∂θ_g
      grad = g.gradient(g_loss, self.model.trainable_variables)

      # Updating θ_g := θ_g - α(∂L_g/∂θ_g)
      self.gan.generative_optimizer.apply_gradients(zip(grad, self.model.trainable_variables))