'''
To understand these variables, please visit gan_tutorial.ipynb!
'''
import tensorflow as tf
import gc

class GAN():
  def __init__(self, 
               batch_size, 
               window_length,
               generative_optimizer, 
               discriminative_optimizer,
               generative_total_iterations,
               discriminative_total_iterations,
               generative_latent_sample_size,
               generative_latent_sample_mean,
               generative_latent_sample_stdev):
    self.batch_size = batch_size
    self.window_length = window_length
    self.generative_optimizer = generative_optimizer
    self.discriminative_optimizer = discriminative_optimizer
    self.generative_total_iterations = generative_total_iterations
    self.discriminatvie_total_iterations = discriminative_total_iterations
    self.generative_latent_sample_size = generative_latent_sample_size
    self.generative_latent_sample_mean = generative_latent_sample_mean
    self.generative_latent_sample_stdev = generative_latent_sample_stdev
    pass

  # to free up unused memory (garbage)
  def free_garbage(self):
    gc.collect()

  # to ease CPU/GPU usage (keras backend session)
  def free_keras_session(self):
    tf.keras.backend.clear_session()

