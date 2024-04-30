import tensorflow as tf
import tensorflow.math as tfm
import my_notebook_modules as mynbm
from tqdm import tqdm

def write_and_analyze(ds__, out_file__, batch_size__, 
                      take_size__, cum_batch__, keys__):
  cums = [[], [], [], [], []]
  analysis_matrix = [[], [], [], []]
  cum_data = [0 for _ in keys__]
  cum_batch = 0
  write_incidents = 0
  bar = tqdm(total=1, position=0, bar_format='[{elapsed}] {desc}')

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
    bar.set_description_str(f'Batch rounds: {cum_batch}')
    
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
      out_file__ + f'-{write_incidents+1}.tfr')

  for i, cum in enumerate(cums[:-1]):
    my_cum = tf.convert_to_tensor(cum)
    my_cum = tf.reshape(tf.transpose(my_cum, perm=[2,0,1]), shape=(2,-1))
    analysis_matrix[i].append(tfm.reduce_min(my_cum, axis=1))
    analysis_matrix[i].append(tfm.reduce_max(my_cum, axis=1))
    analysis_matrix[i].append(tfm.reduce_mean(my_cum, axis=1))
    analysis_matrix[i].append(tfm.reduce_std(my_cum, axis=1))
    
  analysis_matrix = [analysis_matrix, cums[0], cums[1], cums[2], 
                     cums[3], cums[4]]
  mynbm.dataset_utils.io.write_tfr_from_list(analysis_matrix,
                                            ['analysis.f32', 'min.f32',
                                            'max.f32', 'avg.f32', 'std.f32',
                                            'dist.f32'],
                                             out_file__ + '-analysis.tfr')
  bar.close()
  return tf.convert_to_tensor(analysis_matrix[0])