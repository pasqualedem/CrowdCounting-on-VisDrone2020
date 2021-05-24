import numpy as np
import pandas as pd
import os
import torch
import torchvision
from PIL import Image as pil
import h5py

SIZE = ()
FILE_EXTENSION = '.jpg'
GT_FILE_EXTENSION = '.h5'
LOG_PARA = 2550.0


def scale(x):
    return torch.from_numpy(np.array(x)) * LOG_PARA


class VisDroneDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, img_transform=True, gt_transform=True):
        self.dataframe = dataframe

        self.mean = [0.43476477, 0.44504763, 0.43252817]
        self.std = [0.20490805, 0.19712372, 0.20312176]
        self.img_transform = None
        self.gt_transform = None
        if img_transform:
            # Initialize data transforms
            self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize(mean=self.mean,
                                                                                              std=self.std),
                                                             ])  # normalize to (-1, 1)
        if gt_transform:
            self.gt_transform = torchvision.transforms.Compose([
                scale
            ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        # Obtain the filename and target
        filename = self.dataframe.loc[i]['filename']
        target_filename = self.dataframe.loc[i]['gt_filename']

        # Load and transform the image
        with pil.open(filename) as img:
            if self.img_transform:
                data = self.img_transform(img)
            else:
                data = np.array(img)

        hf = h5py.File(target_filename, 'r')
        target = hf.get('density')

        if self.gt_transform:
            target = self.gt_transform(target)

        return data, torch.sum(target)

    def get_targets(self):
        return self.targets


def make_dataframe(folder):
    # Return a DataFrame with columns (example folder, example idx, filename, gt filename)
    folders = os.listdir(folder)
    dataset = []
    for cur_folder in folders:
        files = os.listdir(os.path.join(folder, cur_folder))
        for file in files:
            if FILE_EXTENSION in file:
                idx, ext = file.split('.')
                gt = os.path.join(folder, cur_folder, idx + GT_FILE_EXTENSION)
                dataset.append([idx, os.path.join(folder, cur_folder, file), gt])
    return pd.DataFrame(dataset, columns=['id', 'filename', 'gt_filename'])


def load_test():
    df = make_dataframe('../dataset/VisDrone2020-CC/sequences')
    ds = VisDroneDataset(df)
    return ds