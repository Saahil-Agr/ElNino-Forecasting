import random
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import torch




# loader for training (maybe apply data augmentation, not for now)
train_transformer = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()])

# loader for validation
val_transformer = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()])


class ClimateDataset(Dataset):

    def __init__(self, data_dir, transform):

        self.images_path = os.path.join(data_dir, "{}".format('img_not_scaled'))
        self.labels_path = os.path.join(data_dir, "{}".format('labels.csv'))

        self.filenames = os.listdir(self.images_path)
        self.filenames = [os.path.join(self.images_path, f) for f in self.filenames if f.endswith('.npy')]
        labels_df = pd.read_csv(self.labels_path)
        self.labels = np.asarray(labels_df['tas'].tolist())
        print('number of labels',len(self.labels))
        print('number of inputs', len(self.filenames))

        # save the tensor transform
        self.transform = transform
        # load all imputs
        self.inputs = [np.load(self.filenames[idx]) for idx in range(len(self.filenames))]
        self.inputs = [input.reshape(input.shape[2], input.shape[0], input.shape[1]) for input in self.inputs]
        self.inputs = [torch.from_numpy(input) for input in self.inputs]

        # I put in comments the transformation we used for images.
        #self.inputs = [input.reshape(input.shape[0], input.shape[1]) for input in self.inputs]
        #self.inputs = [Image.fromarray(input, mode = 'L') for input in self.inputs]
        #self.inputs = [self.transform(input) for input in self.inputs]


    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):

        return self.inputs[idx], self.labels[idx]  # return both as tensors


def fetch_dataloader(types, data_dir, batch_size):

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:

            path = os.path.join(data_dir, "{}".format(split))

            if split == 'train':
                print('loading training set')
                dl = DataLoader(ClimateDataset(path, train_transformer), batch_size=batch_size, shuffle=True)
            else:
                print('loading validation set')
                dl = DataLoader(ClimateDataset(path, val_transformer), batch_size=batch_size, shuffle=False)

            dataloaders[split] = dl

    return dataloaders
