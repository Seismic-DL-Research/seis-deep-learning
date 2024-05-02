import tensorflow as tf
import my_notebook_modules as mynbm
import my_notebook_modules.dataset_utils.op.dataset_operation as mynbm_dsop

def distribute_data(dataset__, dataset_keys__, take_size__, batches_per_file__,
hist__, hist_length__, key__):
  curr_val = 0
  frequencies, bins = hist__
  
  for i in range(0, len(bins)):
    curr_val += hist_length__
    uniformed_ds = dataset__.filter(mynbm_dsop.filterFunc_specific_key_range(
      bot_range__=curr_val - hist_length__,
      top_range__=curr_val,
      key=key__
    ))

    out_file__ += f'_{key__}_from_{curr_val - hist_length__:.2f}_to_{curr_val + hist_length__:.2f}'
    mynbm.dataset_utils.io.write_tfr_from_dataset(
      ds__=dataset__, 
      keys__=dataset_keys__,
      take_size__=take_size__,
      out_file__=out_file__,
      batches_per_file__=batches_per_file__
    )

  pass
