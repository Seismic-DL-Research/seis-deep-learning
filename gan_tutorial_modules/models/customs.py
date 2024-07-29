'''
Message from notebook "gan_tutorial.ipynb":
The input layer is a vector in R^L space.
If previously we only pass shape=(50,), this time we need to assign the parameter
shape of the input layer with (50,1). Although mathematically this also means a
vector, TensorFlow regards this as a "channel". This is the requirement for
passing the tensor into a Conv1DTranspose layer.

Alternatively, we can still maintain shape=(50,) as the input layer, but we need
to use the following customized layer to expand it.
'''
import tensorflow as tf

# -- CUSTOM LAYER
@tf.keras.utils.register_keras_serializable(
    package="gan-tutorial",
    name="VecAddChannel",
)
class VecAddChannel(tf.keras.Layer):
  def __init__(sf):
    # The method called when the class's object is initiated.

    # We call the parent's constructor
    super().__init__()

  def build(sf, input_shape):
    # This method usually contains trainable weights. We won't add
    # any for this layer.
    pass

  def call(sf, input):
    # We returned its expanded dims form.
    return tf.expand_dims(input, axis=-1) 

# -- CUSTOM ACTIVATION FUNCTION
@tf.keras.utils.register_keras_serializable(
    package="gan-tutorial",
    name="ab_leaky_relu",
)
def ab_leaky_relu(a, b):
  def proc(x):
    # x is a tensor. The process is element-wise.

    # We obtain all the negative values and its corresponding indices in the
    # tensor.
    negative_values = tf.boolean_mask(x, x < 0)
    negative_values_idc = tf.where(x < 0)

    # We obtain all the positive values and its corresponding indices in the
    # tensor.
    positive_values = tf.boolean_mask(x, x >= 0)
    positive_values_idc = tf.where(x >= 0)

    # We multiply all of the negative values by 'a' and all of the positive
    # values by 'b'
    negative_values *= a
    positive_values *= b

    # update all of the negative values
    new_x = tf.tensor_scatter_nd_update(tensor=x, 
                                        indices=negative_values_idc,
                                        updates=negative_values)

    # update all of the positive values
    new_x = tf.tensor_scatter_nd_update(tensor=new_x, 
                                        indices=positive_values_idc,
                                        updates=positive_values)

    return new_x
  return proc