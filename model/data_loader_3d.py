import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch


# # loader for training (maybe apply data augmentation, not for now)
train_transformer = transforms.Compose([transforms.ToTensor()])
#     transforms.Resize((64,64)),
#
#
# # loader for validation
val_transformer = transforms.Compose([transforms.ToTensor()])
#     transforms.Resize((64,64)),
#     transforms.ToTensor()])


class ClimateDataset(Dataset):

    def __init__(self, data_dir, transform):

        self.images_path = os.path.join(data_dir, "{}".format('full_data'))
        self.labels_path = os.path.join(data_dir, "{}".format('full_data/labels.csv'))

        self.filenames = os.listdir(self.images_path)
        self.filenames = [os.path.join(self.images_path, f) for f in self.filenames if f.endswith('.npy')]

        labels_df = pd.read_csv(self.labels_path)
        self.labels_time = np.asarray(labels_df['time'].tolist())
        self.labels = np.asarray(labels_df['tas'].tolist())
        print('number of labels', len(self.labels))
        print('number of inputs', len(self.filenames))
        self.transform = transform

        self.inputs = [np.load(self.filenames[idx]) for idx in range(len(self.filenames))]
        self.inputs = [np.expand_dims(input, axis = 0) for input in self.inputs]
        #print(self.inputs.shape)
        #self.inputs = [input.reshape(input.shape[2], input.shape[0], input.shape[1]) for input in self.inputs]
        self.inputs = [torch.from_numpy(input) for input in self.inputs]

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):

        #array = np.load(self.filenames[idx])
        #print(type(array))
        #array1 = np.expand_dims(array, axis = 0)
        #print(type(array))
        #print("modified",array1.shape)
        #array = array.reshape(array.shape[0], array.shape[1])
        #image = Image.fromarray(array, mode = 'L') # PIL image
        #array = torch.from_numpy(array1)  #transform to tensor
        #print(self.labels[idx + 24])
        #print(self.inputs[idx].shape)
        return self.inputs[idx], self.labels[idx]# return both as tensors return labels at 48th location


def fetch_dataloader(types, data_dir, batch_size):

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:

            path = os.path.join(data_dir, "{}".format(split))

            if split == 'train':
                dl = DataLoader(ClimateDataset(path, train_transformer), batch_size=batch_size, shuffle=True)
            else:
                dl = DataLoader(ClimateDataset(path, val_transformer), batch_size=batch_size, shuffle=False)

            dataloaders[split] = dl

    return dataloaders
