import shutil
import tarfile as Tar
import my_notebook_modules as mynbm
import os

def fetch_extract_and_list(THESISPATH__, identifier__):
  mynbm.cwd.non_git()

  modes = ['UD', 'EW', 'NS']
  for mode in modes:
    target_file = f'{THESISPATH__}/{mode}/'
    gz_filename = f'{mode}1.{mode}2.{identifier__}.gz'
    temp_folder = 'temp1-' + identifier__.split('.')[0][5:]

    shutil.copy(target_file + gz_filename, '.')
    with Tar.open(gz_filename, 'r') as z:
      z.extractall(f'{mode}')

  mynbm.cwd.git()
  return [i.split('.')[0]
          for i in os.listdir(f'{mode}/{temp_folder}')], temp_folder