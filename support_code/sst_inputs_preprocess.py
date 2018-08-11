import os
import pandas as pd
import numpy as np

'''
script to change all the NaN values to zero and save new file new_img folder
'''

#np.set_printoptions(threshold='nan')

data_dir = '/Volumes/matiascastilloHD/CLIMATEAI/reanalysis/real_scenario_sst/img_without_2018'
new_data_dir = '/Volumes/matiascastilloHD/CLIMATEAI/reanalysis/real_scenario_sst/img_with_mask'
filenames = sorted(os.listdir(data_dir))
print(len(filenames))
filepaths = [os.path.join(data_dir, f) for f in filenames if f.endswith('.npy')]
print(len(filenames))
inputs = [np.load(filepaths[idx]) for idx in range(len(filepaths))]
print(len(inputs))

for i in range(len(inputs)):
    #inputs[i] = np.nan_to_num(inputs[i])
    #inputs[i][inputs[i] > 3] = 0
    #inputs[i][inputs[i] < -3] = 0
    inputs[i][0:47,:,:] = 0
    if i == 10:
        print(inputs[i][:,:,0])
    path = os.path.join(new_data_dir, filenames[i])
    np.save(path, inputs[i])
