import tensorflow as tf
import tensorflow.math as tfm
import tensorflow.linalg as tfl

@tf.keras.utils.register_keras_serializable(
    package="thesis-cvnn",
    name="complex_residual_2d"
)
class complex_residual_2d(tf.keras.layers.Layer):
  def __init__(sf):
    super(complex_residual_2d, sf).__init__()

  def construct_matrix(sf, shape__, name__):
    '''
      construct_matrix:
        * called to construct trainable weights in the model.
      receive:
        * shape__ (input shape)
        * name__ (kernel name)
      return:
        * matrix (shape: depends on "shape__")
    '''
    matrix = sf.add_weight(
        shape=shape__,
        initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
        trainable=True,
        name=name__
    )
    return matrix

  def to_nchw(sf, matrix__):
    '''
      to_nchw
        * used to convert NHWC tensor format into NCHW format via tf.transpose with
          perm [0,3,1,2]
      receive:
        * shape__ (input shape | shape: NxHxWxC)
        * name__ (kernel name)
      return:
        * transposed matrix (shape: NxCxHxW)
    '''
    # matrix__: N x H x W x C
    return tf.transpose(matrix__, perm=[0,3,1,2])

  def to_nhwc(sf, matrix__):
    '''
      to_nchw
        * used to convert NHWC tensor format into NCHW format via tf.transpose with
          perm [0,3,1,2]
      receive:
        * shape__ (type: tuple | input shape | shape: NxCxHxW)
        * name__ (type: tuple | kernel name)
      return:
        * transposed matrix (shape: NxHxWxC)
    '''
    # matrix__: N x C x H x W
    return tf.transpose(matrix__, perm=[0,2,3,1])

  def consecutive_matmul(sf, matrix_sets__):
    '''
      consecutive_matmul
        * Used to weigh matrix operations from the given tuples of two matrices.
        * On each tuple, tfl.matmul operation is conducted. 
        * All the result of tfl.matmul are added element-wise-ly.
      receive:
        * "matrix_sets" (type: list | containing tuples of two matrices)
      return:
        * complex matrix (shape: NxHxWxC)
      WARNING:
        * make sure that all matrices in the tuples are same-shaped! Element-wise
        addition will be conducted!
    '''
    sum_matrix = 0
    for matrix_set in matrix_sets__:
      sum_matrix += tf.matmul(matrix_set[0], matrix_set[1])
    return sum_matrix

  def build(sf, input_shape__):
    # alpha_shape: (N, H0, W0, C0)
    alpha_shape = input_shape__[1]
    # beta_shape: (N, H1, W1, C0)
    beta_shape = input_shape__[0]

    tf.print(alpha_shape)
    tf.print(beta_shape)

    # P_shape: N x C0 x H1 x H0 (in NCHW format)
    P_shape = (1, alpha_shape[-1], beta_shape[1], alpha_shape[1])
    sf.PR = sf.construct_matrix(P_shape, 'P_matrix_real')
    sf.PJ = sf.construct_matrix(P_shape, 'P_matrix_imag')

    # Q_shape: N x C0 x W0 x W1 (in NCHW format)
    Q_shape = (1, alpha_shape[-1], alpha_shape[2], beta_shape[2])
    sf.QR = sf.construct_matrix(Q_shape, 'Q_matrix_real')
    sf.QJ = sf.construct_matrix(Q_shape, 'Q_matrix_imag')

  def call(sf, input__):
    alpha__ = sf.to_nchw(input__[1])
    beta__ = sf.to_nchw(input__[0])

    alphaR = tfm.real(alpha__)
    alphaJ = tfm.imag(alpha__)
    betaR = tfm.real(beta__)
    betaJ = tfm.imag(beta__)

    u = sf.consecutive_matmul([(alphaR, sf.QR), (-1 * alphaJ, sf.QJ)])
    v = sf.consecutive_matmul([(alphaR, sf.QJ), (alphaJ, sf.QR)])

    PalphaQR = sf.consecutive_matmul([(sf.PR, u), (-1 * sf.PJ, v)])
    PalphaQJ = sf.consecutive_matmul([(sf.PR, v), (sf.PJ, u)])
    PalphaQ = tf.complex(PalphaQR, PalphaQJ)

    return beta__ + PalphaQ