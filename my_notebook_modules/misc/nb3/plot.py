import matplotlib.pyplot as plt
import tensorflow.math as tfm
import seaborn as sns
import numpy as np
import scipy

def heatmap(analysis_matrix__):
  f, ax = plt.subplots(ncols=2, figsize=(12,5))
  sns.heatmap(analysis_matrix__[:,:,0],
              cmap='coolwarm',
              annot=True,
              xticklabels=['min', 'max', 'avg', 'std'],
              yticklabels=['min', 'max', 'avg', 'std'],
              linecolor='black',
              linewidths='0.05',
              ax=ax[0],
              square=True,
              fmt='.05f')
  sns.heatmap(analysis_matrix__[:,:,1],
              cmap='coolwarm',
              annot=True,
              xticklabels=['min', 'max', 'avg', 'std'],
              yticklabels=['min', 'max', 'avg', 'std'],
              linecolor='black',
              linewidths='0.05',
              square=True,
              ax=ax[1],
              fmt='.05f')
  ax[0].set_title('Real Values Matrix')
  ax[1].set_title('Imaginary Values Matrix')
  plt.show()

def distance_dist(distances):
  f, ax = plt.subplots(ncols=2, figsize=(12,5))
  b = scipy.stats.gaussian_kde(distances)
  plt.tight_layout(pad=3)

  x = np.arange(0, 300, .1)
  ax[0].set_title('Histogram')
  ax[0].hist(distances, bins=20, color='black')
  ax[0].set_ylabel('Frequency')
  ax[1].set_title('KDE Dist.')
  ax[1].set_xlabel('Distances')
  ax[1].plot(x, b(x), color='black', label='KDE graph of distance distribution')
  ax[1].axvline(tfm.reduce_mean(distances), color='red', label='Mean')
  ax[1].set_ylabel('Density')
  ax[1].set_xlabel('Distances')
  ax[1].legend()

  plt.show()