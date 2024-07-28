'''
WARNING! This model is just for a tutorial purpose!
If you want to actually train your data, you may want to hyper tune or
modify this model. Please act according to your necessity.
'''
import tensorflow as tf
import gan_tutorial_modules as gtm
from gan_tutorial_modules.models import GAN
from tensorflow.keras.layers import Dense, Conv1DTranspose, Input, Flatten, Conv1D, \
AveragePooling1D

class Discriminative(GAN):
  def __init__(self, gan):
    gan.discriminative_module = self
    self.gan = gan

    # discriminative model
    input_layer = Input(shape=(self.gan.window_length,))
    expanded = gtm.models.customs.VecAddChannel()(input_layer)

    # load custom Leaky ReLU
    ab_leaky_relu = gtm.models.customs.ab_leaky_relu

    # begin convolution
    cont = Conv1D(filters=3, kernel_size=5, activation=ab_leaky_relu(0.1,1))(expanded)
    cont = Conv1D(filters=4, kernel_size=8, activation=ab_leaky_relu(0.1,1))(cont)
    cont = Conv1D(filters=3, kernel_size=10, activation=ab_leaky_relu(0.1,1))(cont)
    cont = AveragePooling1D(pool_size=2)(cont)

    # exit convolution, begin fully-connected
    cont = Flatten()(cont)
    cont = Dense(256, activation=ab_leaky_relu(0.1,1))(cont)
    cont = Dense(128, activation=ab_leaky_relu(0.1,1))(cont)
    cont = Dense(64, activation=ab_leaky_relu(0.1,1))(cont)
    cont = Dense(32, activation=ab_leaky_relu(0.1,1))(cont)
    cont = Dense(1, activation='sigmoid')(cont)

    # construct the model
    self.model = tf.keras.Model(inputs=[input_layer], outputs=[cont])

  
  @tf.function # adding decorator to accelerate training later
  def loss(self, X, X_prime):
    # Variable X and X_prime are R^(BxN) with B is the batch size.
    
    # define epsilon to avoid nan
    eps = 1e-6
    
    # first term:
    first_term = tf.math.log(1 - self.model(X_prime) + eps) # shape: B x 1

    # second term:
    second_term = tf.math.log(self.model(X) + eps) # shape: B x 1

    # total loss
    loss = first_term + second_term # shape: B x 1
    loss = tf.math.reduce_sum(loss, axis=0) # shape: 1
    loss = - 1* loss[0] # scalar

    return loss

  @tf.function
  def update_trainable_tensors(self, X, X_prime):
    # X is P-wave data (shape: BxN)
    # X_prime is non-P-wave data (shape: BxN)

    with tf.GradientTape() as d:
      # calculating L_d
      d_loss = self.loss(X, X_prime)

      # calculating ∂L_d/∂θ_d
      grad = d.gradient(d_loss, self.model.trainable_variables)

      # updating θ_d := θ_d - α(∂L_d/∂θ_d)
      self.gan.discriminative_optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

  def update_model(self, new_model):
    contains_error = False

    # check if model's I/O shape is valid
    input_shape = new_model.layers[0].output.shape[1]
    output_shape = new_model.layers[-1].output.shape[1]

    try:
      assert input_shape == self.gan.window_length
    except:
      print(f'Invalid input shape! Expected {self.gan.window_length} but obtained {input_shape}')
      contains_error = True
    
    try:
      assert output_shape == 1
    except:
      print(f'Invalid output shape! Expected 1 but obtained {output_shape}')
      contains_error = True

    # update model if valid
    if not contains_error: self.model = new_model