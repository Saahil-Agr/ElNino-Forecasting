import numpy as np
import os
import pandas as pd
#import torchvision.transforms as transforms
#from PIL import Image

data_dir = "data/scaled"
filename = os.listdir(data_dir)
print(len(filename))
filenames = [os.path.join(data_dir, f) for f in filename[:430] if f.endswith('.npy')]
train_dir = "data/3d/train/small_data"
val_dir = "data/3d/val/small_data"
test_dir = "data/3d/test"
labels_path = os.path.join(data_dir, "{}".format('labels.csv'))
labels_df = pd.read_csv(labels_path)
print(filenames[-1])
# required parameters
noOf_files = len(filenames)
span = 24
new_No_files = noOf_files - span + 1
train_no = 300
val_no = 50
#test_no = 50

test_file = np.load(filenames[0])
H,W = test_file.shape[0], test_file.shape[1]
stacked_files = np.empty((noOf_files, H, W))

labels_df[span:train_no+span].to_csv(train_dir + "/labels.csv")
labels_df[train_no+span : train_no+span +val_no].to_csv(val_dir + "/labels.csv")
labels_df[train_no+span + val_no :noOf_files].to_csv(test_dir + "/labels.csv")

#transformer = transforms.Compose([transforms.Resize((64,64))])
for idx,f in enumerate(filenames):
    temp_file = np.load(f).reshape((H,W))
    #image = Image.fromarray(temp_file,mode = 'L')
    #image = transformer(image)
    #temp_file_resized = np.asanyarray(image, dtype = temp_file.dtype)
    stacked_files[idx] = temp_file
    if idx >= span:
        if idx < (train_no + span):
            f_name = f.split('.')[0].split('/')[-1] + '_3d'
            f_path = os.path.join(train_dir, f_name)
            np.save(f_path, stacked_files[idx-span:idx])
            #fp = os.path.join(write_dir, label)
        elif idx < (train_no + span + val_no):
            f_name = f.split('.')[0].split('/')[-1] + '_3d'
            f_path = os.path.join(val_dir, f_name)
            np.save(f_path, stacked_files[idx - span:idx])
        else:
            f_name = f.split('.')[0].split('/')[-1] + '_3d'
            f_path = os.path.join(test_dir, f_name)
            np.save(f_path, stacked_files[idx - span:idx])