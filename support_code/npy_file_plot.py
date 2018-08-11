import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
script to plot a .npy image with matplotlib. This script is useful to compare the .npy images 
with the original netCDF files (.nc) to check if the images are the same.
'''

path = '/Volumes/matiascastilloHD/CLIMATEAI/GCMs/HadGEM2-ES/piControl_r1i1p1/tas_images/tas_Amon_HadGEM2-ES_piControl_r1i1p1_1859-12.npy'
image = np.load(path)
print(image.shape)
print(image)
image = np.squeeze(image)
plt.imshow(image, cmap='hot')
plt.colorbar()
plt.show()



