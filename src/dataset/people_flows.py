import os
import torch
import torchvision
import h5py

from PIL import Image as pil
import numpy as np
import pandas as pd

ROOT_DATAPATH = '../dataset/FDST'
TRAIN_DATAPATH = os.path.join(ROOT_DATAPATH, 'train_data')
TEST_DATAPATH = os.path.join(ROOT_DATAPATH, 'test_data')
PREV_FRAME_DIFFERENCE = 5
SIZE = (640, 360)
FILE_EXTENSION = '.jpg'


class PeopleFlowsDataset(torch.utils.data.Dataset):
    """
    Torch dataset subclass for loading the PeopleFlows dataset
    """
    def __init__(self, path, dataframe):
        self.path = path

        self.dataframe = dataframe
        # Initialize data transforms
        self.transform = transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(SIZE),
            torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225]),
        ])# normalize to (-1, 1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        # Obtain the filename and target
        filename = self.dataframe.loc[i]['filename']
        target_name = self.dataframe.loc[i]['gt_filename']
        frame_id = int(self.dataframe.loc[i]['id'])

        prev_id = max(1, frame_id - PREV_FRAME_DIFFERENCE)
        if prev_id == 1:
            prev_frame_diff = i - frame_id + 1
        else:
            prev_frame_diff = PREV_FRAME_DIFFERENCE

        prev_filename = self.dataframe.loc[i - prev_frame_diff]['filename']

        # Load and transform the image
        with pil.open(filename) as img:
            data = self.transform(img)

        with pil.open(prev_filename) as prev_img:
            prev_data = self.transform(prev_img)

        gt_file = h5py.File(target_name)
        target = np.asarray(gt_file['density'])

        return data, prev_data, np.sum(target)

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
                gt = os.path.join(folder, cur_folder, idx + '_resize.h5')
                dataset.append([cur_folder, idx, os.path.join(folder, cur_folder, file), gt])
    return pd.DataFrame(dataset, columns=['folder', 'id', 'filename', 'gt_filename'])
