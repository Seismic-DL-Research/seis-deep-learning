import tensorflow as tf
import tensorflow.math as tfm
import my_notebook_modules as mynbm

def write_and_analyze(ds__, out_file__, batch_size__, 
                      take_size__, cum_batch__, keys__):
  cums = [[], [], [], [], []]
  analysis_matrix = [[], [], [], []]
  cum_data = [0 for _ in keys__]
  cum_batch = 0
  write_incidents = 0
  for m, data_elem in enumerate(ds__.batch(batch_size__, drop_remainder=True)
                                .take(take_size__)):
    for i, stat in enumerate(mynbm.misc.nb3.get_stat(data_elem['data'])): 
      cums[i].append(stat)
    cums[-1].append(data_elem['dist'])

    for i, key in enumerate([i.split('.')[0] for i in keys__]):
      if type(cum_data[i]) == int:
        cum_data[i] = data_elem[key]
      else:
        cum_data[i] = tf.concat([cum_data[i], data_elem[key]], axis=0)

    cum_batch += 1

    if cum_batch %  cum_batch__ == 0:
      write_incidents += 1
      mynbm.dataset_utils.io.write_tfr_from_list(
        cum_data, 
        keys__, 
        out_file__ + f'-{write_incidents}.tfr')
      cum_data = [0 for _ in keys__]
  #  
  if (type(cum_data[0]) != int):
    mynbm.dataset_utils.io.write_tfr_from_list(
      cum_data, 
      keys__, 
      out_file__ + f'-{write_incidents}.tfr')

  for i, cum in enumerate(cums[:-1]):
    my_cum = tf.convert_to_tensor(cum)
    my_cum = tf.reshape(tf.transpose(my_cum, perm=[2,0,1]), shape=(2,-1))
    analysis_matrix[i].append(tfm.reduce_min(my_cum, axis=1))
    analysis_matrix[i].append(tfm.reduce_max(my_cum, axis=1))
    analysis_matrix[i].append(tfm.reduce_mean(my_cum, axis=1))
    analysis_matrix[i].append(tfm.reduce_std(my_cum, axis=1))
    
  analysis_matrix = [analysis_matrix]
  mynbm.dataset_utils.io.write_tfr_from_list(analysis_matrix,
                                            ['analysis.f32'],
                                             out_file__ + '-analysis.tfr')

  return tf.convert_to_tensor(analysis_matrix[0])