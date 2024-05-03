import tensorflow as tf
import tensorflow.math as tfm
import my_notebook_modules as mynbm
from tqdm import tqdm

def trainer(model__, train_dataset__, valid_dataset__, opt__, batch_size__, epoch__, take_num__):
  total_batch = 0

  for epoch in range(1, epoch__ + 1):
    total = 1 if epoch == 1 else total_batch
    # total_loss = []
    print(f'\n\U0001f534 Epoch {epoch} out of {epoch__}')
    bar = tqdm(total=total, ascii='_█', position=0,
               bar_format='|{bar:30}| [{elapsed}<{remaining}] {desc}')

    for i, train_dataset in enumerate(train_dataset__.batch(batch_size__).take(take_num__)):
      with tf.GradientTape() as g:
        # get model's epicentral distance estimation values
        data = train_dataset['data']
        y_hat = model__(data)
        y = tf.expand_dims(train_dataset['dist'] / 10, axis=0)
        
        # calculating the loss
        loss = mynbm.model.epicenter.mae(y_hat, y)

        # apply gradient descent to update weights
        grad = g.gradient(loss, model__.trainable_variables)
        opt__.apply_gradients(zip(grad, model__.trainable_variables))

      if epoch == 1:
        total_batch += 1
        bar.set_description_str(f'Loss: {loss:.4f}')
      else:
        bar.update(1)
        bar.set_description_str(f'Batch {i}/{total_batch} | Loss: {loss:.4f}')
    
    val_losses = 0
    for i, valid_dataset in enumerate(valid_dataset__.batch(batch_size__).take(-1)):
      predicted = model__(valid_dataset['data'])
      real = valid_dataset['data']

      val_loss = mynbm.model.epicenter.mae(predicted, real)
      val_losses += val_loss 

    val_loss = val_losses / (i+1)
    
    print(f'Validation Loss: {val_loss:.4f}')


    #   total_loss.append(float(loss))
    # print('Train Avg Loss: ', float(tfm.reduce_mean(tf.convert_to_tensor(total_loss))))

    bar.close()