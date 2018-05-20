import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch



images_path = os.path.join('data/', "{}".format('img'))
#labels_path = os.path.join('data/train/', "{}".format('labels_small.csv'))

filenames = os.listdir(images_path)
filenames = [os.path.join(images_path, f) for f in filenames if f.endswith('.npy')]
#labels_df = pd.read_csv(labels_path)

#labels = np.asarray(labels_df['tas'].tolist())
#print('labels',len(labels))
print(len(filenames))

inputs = [np.load(filenames[idx]) for idx in range(len(filenames))]

print(len(inputs))
print(inputs[0].shape)
inputs = [input.reshape(input.shape[0], input.shape[1]) for input in inputs]


#inputs = [Image.fromarray(input, mode = 'L') for input in inputs]


from scipy import misc
#misc.imsave('outfile.jpg', inputs[0])

image = misc.imread('outfile.jpg')

print(image)


print(inputs[0])
#inputs = [Image.fromarray(input, mode = 'L') for input in inputs]
#inputs = [transform(input) for input in inputs]
