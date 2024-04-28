import tensorflow as tf
import my_notebook_modules as mynbm

def trainer(model__, train_dataset__, opt__, batch_size__, epoch__):
  for epoch in range(epoch__):
    for train_dataset in train_dataset__.batch(batch_size__).take(-1):
      with tf.GradientTape() as g:
        # get model's epicentral distance estimation values
        y_hat = model__(train_dataset['data'][:,:,:,:50,:])
        y = tf.expand_dims(train_dataset['dist'], axis=0)
        
        # calculating the loss
        loss = mynbm.model.epicenter.mae(y_hat, y)

        # apply gradient descent to update weights
        grad = g.gradient(loss, model__.trainable_vars)
        opt__.apply_gradients(zip(grad, model__.trainable_vars))
      print('Loss: ', loss)
  pass