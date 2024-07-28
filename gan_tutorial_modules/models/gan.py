'''
To understand these variables, please visit gan_tutorial.ipynb!
'''
import tensorflow as tf
import gc

class GAN():
  def __init__(self, 
               epoch,
               batch_size, 
               window_length,
               generative_optimizer, 
               discriminative_optimizer,
               generative_total_iterations,
               discriminative_total_iterations,
               generative_latent_sample_size,
               generative_latent_sample_mean,
               generative_latent_sample_stdev,
               n_wave_dataset,  # stands for Pre-P phase wave (nosie)
               p_wave_dataset # stands for P-wave
              ):
    self.batch_size = batch_size
    self.epoch = epoch
    self.window_length = window_length
    self.generative_optimizer = generative_optimizer
    self.discriminative_optimizer = discriminative_optimizer
    self.generative_total_iterations = generative_total_iterations
    self.discriminative_total_iterations = discriminative_total_iterations
    self.generative_latent_sample_size = generative_latent_sample_size
    self.generative_latent_sample_mean = generative_latent_sample_mean
    self.generative_latent_sample_stdev = generative_latent_sample_stdev
    self.discriminative_module = -1
    self.generative_module = -1
    self.p_wave_dataset = p_wave_dataset.unbatch()
    self.n_wave_dataset = n_wave_dataset.unbatch()
    pass

  # to free up unused memory (garbage)
  def free_garbage(self):
    gc.collect()

  # to ease CPU/GPU usage (keras backend session)
  def free_keras_session(self):
    tf.keras.backend.clear_session()
  
  # train GAN model
  def train(self):
    batched_p_wave = self.p_wave_dataset.batch(self.batch_size)
    batched_n_wave = self.n_wave_dataset.batch(self.batch_size)
    total_batches = 0

    for epoch in range(1, self.epoch+1):
      print(f'Epoch {epoch} out of {self.epoch}')

      for p, n in zip(batched_p_wave.take(-1), batched_n_wave.take(-1)):
        # Counting total batches in the dataset
        if epoch == 1: total_batches += 1
        
        # training generative model
        for _ in range(self.generative_total_iterations): 
          self.generative_module.update_trainable_tensors(self.discriminative_module.model)
          
        # training discriminative model
        for _ in range(self.discriminative_total_iterations):
          # training with P and N from dataset
          self.discriminative_module.update_trainable_tensors(p, n)
          
          # training with P dataset and Generated P (Artificial P)
          latent_sample = tf.random.normal((self.batch_size, self.generative_latent_sample_size),
                                          mean=self.generative_latent_sample_mean,
                                          stddev=self.generative_latent_sample_stdev)

          XG = self.generative_module.model(latent_sample)
          self.discriminative_module.update_trainable_tensors(p, XG)
          

      print(f'Found {total_batches} batches per epoch')


