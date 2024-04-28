import tensorflow as tf
import uuid

@tf.function
def random_name(layer_type__, name__):
  return f'{layer_type__}.{str(uuid.uuid4()).split("-")[0]}'