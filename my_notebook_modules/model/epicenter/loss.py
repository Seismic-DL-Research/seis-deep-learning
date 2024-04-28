import tensorflow as tf
import tensorflow.math as tfm

def mae(y_hat, y):
  # y_hat: N x 1
  abs_err = tfm.abs(y_hat - y)
  mean_abs_err = tfm.reduce_mean(abs_err, axis=0)[0]
  return mean_abs_err

def rmse(y_hat, y):
  diff_square = tfm.square(y_hat - y)
  mean = tfm.reduce_mean(diff_square, axis=0)
  rmse = tfm.sqrt(mean)[0]
  return [0]

# can add more loss function