import numpy as np
import os
import pandas as pd
#import torchvision.transforms as transforms
#from PIL import Image


dtype = np.float16
data_dir = os.path.join('/Volumes/matiascastilloHD/CLIMATEAI/GCMs/MPI-ESM-LR/historical_r1i1p1', "{}".format('tas_images'))
filename = sorted(os.listdir(data_dir))
print(len(filename))
name = 'MPI_hist_tas'
create_labels = True

filenames = [os.path.join(data_dir, f) for f in filename if f.endswith('.npy')]



train_dir = os.path.join('/Volumes/matiascastilloHD/CLIMATEAI/GCMs/MPI-ESM-LR/historical_r1i1p1', "{}".format('3d_tas_images'))
val_dir = os.path.join('/Volumes/matiascastilloHD/CLIMATEAI/GCMs/MPI-ESM-LR/historical_r1i1p1', "{}".format('3d_tas_images'))
test_dir = os.path.join('/Volumes/matiascastilloHD/CLIMATEAI/GCMs/MPI-ESM-LR/historical_r1i1p1', "{}".format('3d_tas_images'))

print(filenames[0])
# required parameters
noOf_files = len(filenames)
print(noOf_files)
span = 24
new_No_files = noOf_files - span + 1 - 6
val_no = 0
print('val', val_no)
test_no = 0
print('test', test_no)
train_no = new_No_files - test_no - val_no
print('train', train_no)
test_file = np.load(filenames[0])
print(test_file.dtype)
H,W = test_file.shape[0], test_file.shape[1]
stacked_files = np.empty((noOf_files, H, W), dtype = dtype)


if create_labels == True:
    labels_path = os.path.join('/Volumes/matiascastilloHD/CLIMATEAI/GCMs/MPI-ESM-LR/historical_r1i1p1', "{}".format('sst_labels.csv'))
    labels_df = pd.read_csv(labels_path)
    labels_df[span:train_no+span].to_csv(train_dir + "/labels.csv")
    labels_df[span+2:train_no+span+2].to_csv(train_dir + "/labels3m.csv")
    labels_df[span+5:train_no+span+5].to_csv(train_dir + "/labels6m.csv")
    #labels_df[train_no+span:train_no+span+val_no].to_csv(val_dir + "/labels.csv")
    #labels_df[train_no+span+2: train_no+span+val_no+2].to_csv(val_dir + "/labels3m.csv")
    #labels_df[train_no+span+5: train_no+span+val_no+5].to_csv(val_dir + "/labels6m.csv")
    #labels_df[train_no+val_no+span:train_no+val_no+test_no+span].to_csv(test_dir + "/labels.csv")
    #labels_df[train_no+val_no+span+2:train_no+span+val_no+test_no+2].to_csv(test_dir + "/labels3m.csv")
    #labels_df[train_no+val_no+span+5:train_no+span+val_no+test_no+5].to_csv(test_dir + "/labels6m.csv")


for idx,f in enumerate(filenames):
    temp_file = np.load(f).reshape((H,W)).astype('float16')
    stacked_files[idx] = temp_file
    if idx >= span:
        name1 = filenames[idx-24].split('_')[-1].split('.')[0]
        name2 = filenames[idx-1].split('_')[-1].split('.')[0]
        if idx < (train_no + span):
            input_name = name + '_from_' + name1 + '_to_' + name2
            f_path = os.path.join(train_dir, input_name)
            np.save(f_path, stacked_files[idx - span:idx])

        elif idx < (train_no + span + val_no):
            input_name = name + '_from_' + name1 + '_to_' + name2
            f_path = os.path.join(val_dir, input_name)
            #np.save(f_path, stacked_files[idx - span:idx])

        elif idx < (train_no + span + val_no + test_no):
            input_name = name + '_from_' + name1 + '_to_' + name2
            f_path = os.path.join(test_dir, input_name)
            #np.save(f_path, stacked_files[idx - span:idx])

    '''
    include last month for evaluation
    
    if idx == len(filenames)-1:
        name1 = filenames[idx-23].split('_')[-1].split('.')[0]
        name2 = filenames[idx].split('_')[-1].split('.')[0]
        input_name = name + '_from_' + name1 + '_to_' + name2
        f_path = os.path.join(train_dir, input_name)
        np.save(f_path, stacked_files[idx-span+1:idx+1])
    '''


