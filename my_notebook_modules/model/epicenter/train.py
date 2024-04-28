import tensorflow as tf
import my_notebook_modules as mynbm

class Train():
  def __init__(sf):
    sf.train_dataset = 0
    sf.model = 0
    sf.opt = 0
    sf.epoch = 0
    sf.batch_size = 0
    pass
  
  def introduce_model(sf, model__):
    sf.model = model__

  def introduce_train_set(sf, train_dataset__):
    sf.train_dataset = train_dataset__
    
  def introduce_optimizer(sf, opt__):
    sf.opt = opt__

  def introduce_epoch(sf, epoch__):
    sf.epoch = epoch__

  def introduce_batch_size(sf, batch_size__):
    sf.batch_size = batch_size__

  def begin(sf):
    mynbm.model.epicenter.trainer(sf.model, sf.train_dataset, 
                                  sf.opt, sf.batch_size, sf.epoch)
