import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image as pil


class VisDroneDataset(torch.utils.data.Dataset):
    def __init__(self, path, dataframe):
        self.path = path
        self.dataframe = dataframe

        self.mean = [0, 0, 0]
        self.std = [1, 1, 1]
        # Initialize data transforms
        self.transform = transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(SIZE),
            torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=self.mean,
                                                                                std=self.std),
        ])# normalize to (-1, 1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        # Obtain the filename and target
        filename = self.dataframe.loc[i]['filename']
        target_filename = self.dataframe.loc[i]['gt_filename']

        # Load and transform the image
        with pil.open(filename) as img:
            data = self.transform(img)

        target = pd.read_csv(target_filename).values

        return data, np.sum(target)

    def get_targets(self):
        return self.targets