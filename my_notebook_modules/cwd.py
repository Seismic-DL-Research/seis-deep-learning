import os

def non_git():
  if (os.getcwd() == '/content'): return
  else: os.chdir('..')

def git():
  if (os.getcwd() == '/content/seis-deep-learning'): return
  else: os.chdir('seis-deep-learning')