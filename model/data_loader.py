import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch


class ClimateDataset(Dataset):

    def __init__(self, data_dir, model_name, split):

        self.images_path = os.path.join(data_dir, "{}".format('full_data_nst'))
        self.labels_path = os.path.join(data_dir, "{}".format('full_data_nst/labels6m.csv'))
        print('im using single channel dataloader')
        print('labels_path: ', self.labels_path)
        self.filenames = sorted(os.listdir(self.images_path))
        self.filenames = [os.path.join(self.images_path, f) for f in self.filenames if f.endswith('.npy')]
        labels_df = pd.read_csv(self.labels_path)

        self.labels = np.asarray(labels_df['skt'].tolist())
        if model_name == 'cnn_many':
            # change labels to arrays of 6 months
            # also need to change getitem function from labels to labels_many
            self.labels_many = np.zeros((len(self.filenames), 6))
            for i in range(len(self.filenames)):
                self.labels_many[i,:] = self.labels[i:i+6]

        print('number of labels',len(self.labels))
        print('number of inputs', len(self.filenames))

        # load all inputs and assign correct dimensions depending on cnn or crnn
        #self.inputs = [np.load(self.filenames[idx]) for idx in range(len(self.filenames))]
        #print(self.inputs[0].shape)
        #if model_name == 'cnn':
         #   self.inputs = [input.reshape(input.shape[2], input.shape[0], input.shape[1]) for input in self.inputs]
        #if model_name == 'crnn' or model_name == 'crnn_many' or model_name == 'crnn_unet':
         #   self.inputs = [np.expand_dims(input, axis = 0) for input in self.inputs]
        #self.inputs = [torch.FloatTensor(input) for input in self.inputs]
        #print(self.inputs[0].shape)

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):

        input = np.load(self.filenames[idx])
        input = torch.FloatTensor(np.expand_dims(input, axis = 0))
        return input, self.labels[idx]  # return both as tensors


def fetch_dataloader(types, data_dir, batch_size, model_name):

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:

            path = os.path.join(data_dir, "{}".format(split))

            if split == 'train':
                print('loading training set')
                dl = DataLoader(ClimateDataset(path, model_name, split), batch_size=batch_size, num_workers=1, shuffle=False)

            elif split == 'val':
                print('loading validation set')
                dl = DataLoader(ClimateDataset(path, model_name, split), batch_size=batch_size, num_workers=1, shuffle=False)

            elif split == 'test':
                print('loading test set')
                dl = DataLoader(ClimateDataset(path, model_name, split), batch_size=batch_size, shuffle=False)

            dataloaders[split] = dl

    return dataloaders
