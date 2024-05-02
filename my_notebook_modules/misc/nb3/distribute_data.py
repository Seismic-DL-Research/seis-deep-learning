import tensorflow as tf
import my_notebook_modules as mynbm
import my_notebook_modules.dataset_utils.op.dataset_operation as mynbm_dsop

def distribute_data(dataset__, dataset_keys__, take_size__, batches_per_file__,
                    hist__, hist_length__, key__, threshold__, tolerance__):
  curr_val = 0
  frequencies, bins = hist__
  
  for i in range(0, len(bins)-2):
    if frequencies[i] < (threshold__ - tolerance__):
      continue
    curr_val += hist_length__
    uniformed_ds = dataset__.filter(mynbm_dsop.filterFunc_specific_key_range(
      bot_range__=bins[i],
      top_range__=bins[i+1],
      key=key__
    )).take(threshold__)

    out_file__ += f'_{key__}_from_{bins[i]:.2f}_to_{bins[i+1]:.2f}'
    mynbm.dataset_utils.io.write_tfr_from_dataset(
      ds__=uniformed_ds, 
      keys__=dataset_keys__,
      take_size__=take_size__,
      out_file__=out_file__,
      batches_per_file__=batches_per_file__
    )
  pass
