'''
To understand these variables, please visit gan_tutorial.ipynb!
'''
import tensorflow as tf
import tqdm
import gc
import copy

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
               n_wave_dataset=None,  # stands for Pre-P phase wave (nosie)
               p_wave_dataset=None, # stands for P-wave
               n_wave_dataset_val=None, # optional
               p_wave_dataset_val=None, # optional
               trigger_threshold=0.9,
               channel='z'
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
    self.trigger_threshold = trigger_threshold

    if channel=='z':
      self.latent_shape = (self.batch_size, self.generative_latent_sample_size)
    elif channel=='zne':
      self.latent_shape = (self.batch_size, self.generative_latent_sample_size, 3)
    else:
      print('WARNING, UNKNOWN CHANNEL!')

    # for debugging
    if p_wave_dataset != None and n_wave_dataset != None:
      self.p_wave_dataset = p_wave_dataset.unbatch().batch(batch_size, drop_remainder=True)
      self.n_wave_dataset = n_wave_dataset.unbatch().batch(batch_size, drop_remainder=True)
    if n_wave_dataset_val != None and p_wave_dataset_val != None:
      self.n_wave_dataset_val = n_wave_dataset_val.unbatch().batch(batch_size)
      self.p_wave_dataset_val = p_wave_dataset_val.unbatch().batch(batch_size)

  # to free up unused memory (garbage)
  def free_garbage(self):
    gc.collect()

  # to ease CPU/GPU usage (keras backend session)
  def free_keras_session(self):
    tf.keras.backend.clear_session()
  
  # train GAN model
  def train(self):
    # updating optimizers
    self.generative_optimizer = copy.deepcopy(self.generative_optimizer)
    self.discriminative_optimizer = copy.deepcopy(self.discriminative_optimizer)
    total_batches = 0

    for epoch in range(1, self.epoch+1):
      tf.keras.backend.clear_session()
      print(f'Epoch {epoch} out of {self.epoch}')

      if epoch != 1:
        bar = tqdm.tqdm(total=total_batches, ascii='._█', position=0,
                        bar_format='|{bar:30}| [{elapsed}<{remaining}] {desc}')
      for p, n in zip(self.p_wave_dataset.take(-1), self.n_wave_dataset.take(-1)):
        
        # Counting total batches in the dataset
        if epoch == 1: total_batches += 1
        else: bar.update(1)

        # training generative model
        for _ in range(self.generative_total_iterations): 
          g_loss = self.generative_module.update_trainable_tensors(self.discriminative_module.model)
          
        # training discriminative model
        for _ in range(self.discriminative_total_iterations):
          # training with P and N from dataset
          _ = self.discriminative_module.update_trainable_tensors(p, n)
          
          # training with P dataset and Generated P (Artificial P)
          latent_sample = tf.random.normal(shape=self.latent_shape,
                                          mean=self.generative_latent_sample_mean,
                                          stddev=self.generative_latent_sample_stdev)

          XG = self.generative_module.model(latent_sample)
          assert tf.shape(p)[0] == tf.shape(XG)[0]
          d_loss = self.discriminative_module.update_trainable_tensors(p, XG)
        
        g_loss = float(g_loss)
        d_loss = float(d_loss)

        if epoch != 1: 
          bar.set_description_str(f'L_G: {g_loss:.4f} | L_D: {d_loss:.4f}')
      
      if epoch != 1: bar.close()

      eval = self.evaluate()

      print(f"LOGS: L_D_val: {eval['d_loss']:.4f} | TP: {eval['true_positive']} | TN: {eval['true_negative']} | FP: {eval['false_positive']} | FN: {eval['false_negative']} \n")
      if epoch == 1: print(f'Found {total_batches} batches per epoch')

  # to evaluate model
  def evaluate(self):
    # total d_loss over the validation data
    total_d_loss = 0
    
    # total validation data
    total_val_data = 0

    for p, n in zip(self.p_wave_dataset_val.take(-1), self.n_wave_dataset_val.take(-1)):
      total_val_data += 1
      total_d_loss += float(self.discriminative_module.loss(p, n))

    # average d loss
    avg_d_loss = total_d_loss / total_val_data

    # False Negative and True Positive
    total_true_positive = 0
    total_false_negative = 0

    for p in self.p_wave_dataset_val:
      predicted_values = self.discriminative_module.model(p)
      total_true_positive += int(tf.shape(tf.where(predicted_values > self.trigger_threshold))[0])
      total_false_negative += int(tf.shape(tf.where(predicted_values < self.trigger_threshold))[0])

    # False Positive and True Negative
    total_false_positive = 0
    total_true_negative = 0

    for n in self.n_wave_dataset_val:
      predicted_values = self.discriminative_module.model(n)
      total_true_negative += int(tf.shape(tf.where(predicted_values < self.trigger_threshold))[0])
      total_false_positive += int(tf.shape(tf.where(predicted_values > self.trigger_threshold))[0])

    return {
      'd_loss': avg_d_loss,
      'true_positive': total_true_positive,
      'false_positive': total_false_positive,
      'true_negative': total_true_negative,
      'false_negative': total_false_negative
    }
