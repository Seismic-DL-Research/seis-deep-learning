import tensorflow as tf

def trainer(model__, train_dataset__, opt__, batch_size__, epoch__):
  for epoch in range(epoch__):
    for train_dataset in train_dataset__.batch(batch_size__).take(-1):
      with tf.GradientTape() as g:
        y_hat = model__(train_dataset['data'][:,:])
  pass