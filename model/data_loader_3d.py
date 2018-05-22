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

        self.images_path = os.path.join(data_dir, "{}".format('img_scaled'))
        self.labels_path = os.path.join(data_dir, "{}".format('labels.csv'))

        self.filenames = os.listdir(self.images_path)
        self.filenames = [os.path.join(self.images_path, f) for f in self.filenames if f.endswith('.npy')]

        labels_df = pd.read_csv(self.labels_path)
        self.labels = np.asarray(labels_df['tas'].tolist())
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):

        array = np.load(self.filenames[idx])
        array = array.reshape(array.shape[0], array.shape[1])
        image = Image.fromarray(array, mode = 'L') # PIL image
        image = self.transform(image)  # resize and transform to tensor
        return image, self.labels[idx]# return both as tensors


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
