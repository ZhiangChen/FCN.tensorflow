import numpy as np
from scipy.io import loadmat

data = loadmat('imagenet-vgg-verydeep-19.mat')

weights = np.squeeze(data['layers'])

p = 1