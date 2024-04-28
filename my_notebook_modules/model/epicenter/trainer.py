import tensorflow as tf
import tensorflow.math as tfm
import my_notebook_modules as mynbm
from tqdm import tqdm

def trainer(model__, train_dataset__, opt__, batch_size__, epoch__):
  total_batch = 0

  for epoch in range(1, epoch__ + 1):
    total = 1 if epoch == 1 else total_batch
    total_loss = []
    print(f'\n\U0001f534 Epoch {epoch} out of {epoch__}')
    bar = tqdm(total=total, ascii='_█', position=0,
               bar_format='|{bar:30}| [{elapsed}<{remaining}] {desc}')

    for i, train_dataset in enumerate(train_dataset__.shuffle(100).batch(batch_size__).take(-1)):
      with tf.GradientTape() as g:
        # get model's epicentral distance estimation values
        data = train_dataset['data'][:,:,:,:50,:]
        max_val = tf.reduce_max(data, axis=-1)
        max_val = tf.reduce_max(max_val, axis=-1)
        max_val = tf.expand_dims(max_val, axis=-1)
        max_val = tf.expand_dims(max_val, axis=-1)
        data = data/max_val
        y_hat = model__(data)
        y = tf.expand_dims(train_dataset['dist'] / 100.0, axis=0)
        
        # calculating the loss
        loss = mynbm.model.epicenter.rmse(y_hat, y)

        # apply gradient descent to update weights
        grad = g.gradient(loss, model__.trainable_variables)
        opt__.apply_gradients(zip(grad, model__.trainable_variables))

      if epoch == 1:
        total_batch += 1
        bar.set_description_str(f'Loss: {loss:.4f}')
      else:
        bar.update(1)
        bar.set_description_str(f'Batch {i}/{total_batch} | Loss: {loss:.4f}')
      total_loss.append(loss)
    print('Train Avg Loss: ', tfm.reduce_mean(tf.convert_to_tensor(loss)))

    bar.close()