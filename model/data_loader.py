import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch


class ClimateDataset(Dataset):

    def __init__(self, data_dir, model_name):

        self.images_path = os.path.join(data_dir, "{}".format('small_data'))
        self.labels_path = os.path.join(data_dir, "{}".format('small_data/labels.csv'))

        self.filenames = sorted(os.listdir(self.images_path))
        self.filenames = [os.path.join(self.images_path, f) for f in self.filenames if f.endswith('.npy')]
        labels_df = pd.read_csv(self.labels_path)
        self.labels = np.asarray(labels_df['tas'].tolist())
        print('number of labels',len(self.labels))
        print('number of inputs', len(self.filenames))

        # load all inputs and assign correct dimensions depending on cnn or crnn
        self.inputs = [np.load(self.filenames[idx]) for idx in range(len(self.filenames))]
        if model_name == 'cnn':
            self.inputs = [input.reshape(input.shape[2], input.shape[0], input.shape[1]) for input in self.inputs]
        if model_name == 'crnn':
            self.inputs = [np.expand_dims(input, axis = 0) for input in self.inputs]
        self.inputs = [torch.FloatTensor(input) for input in self.inputs]


    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):

        return self.inputs[idx], self.labels[idx]  # return both as tensors


def fetch_dataloader(types, data_dir, batch_size, model_name):

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:

            path = os.path.join(data_dir, "{}".format(split))

            if split == 'train':
                print('loading training set')
                dl = DataLoader(ClimateDataset(path, model_name), batch_size=batch_size, shuffle=True)

            else:
                print('loading validation set')
                dl = DataLoader(ClimateDataset(path, model_name), batch_size=batch_size, shuffle=False)


            dataloaders[split] = dl

    return dataloaders
