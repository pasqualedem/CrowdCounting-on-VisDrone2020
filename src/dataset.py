import os
import torch
import torchvision
import pandas as pd
import h5py

from PIL import Image as pil
from PIL import ImageOps as pilops
from sklearn import model_selection
import numpy as np

ROOT_DATAPATH = '../dataset/FDST'
TRAIN_DATAPATH = os.path.join(ROOT_DATAPATH, 'train_data')
TEST_DATAPATH = os.path.join(ROOT_DATAPATH, 'test_data')
PREV_FRAME_DIFFERENCE = 5
SIZE = (640, 360)
FILE_EXTENSION = '.jpg'


class PeopleFlowsDataset(torch.utils.data.Dataset):
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


def load_train_valid_datasets(valid_size=0.1):
    # Load the train dataframe
    filepath = os.path.join(ROOT_DATAPATH, 'train.csv')
    df = pd.read_csv(
        filepath, sep=',', usecols=['filename', 'class'],
        converters={'class': lambda c: 1 if c == 'positive' else 0}
    )

    # Split the dataframe in train and validation
    train_df, valid_df = model_selection.train_test_split(
        df, test_size=valid_size, shuffle=True, stratify=df['class']
    )

    # Instantiate the datasets (notice data augmentation on train data)
    train_data = PeopleFlowsDataset(TRAIN_DATAPATH, train_df)
    valid_data = PeopleFlowsDataset(TRAIN_DATAPATH, valid_df)
    return train_data, valid_data


def load_test_dataset(data_folder=TEST_DATAPATH):
    # Load the test dataframe
    test_df = make_dataframe(data_folder)

    # Instantiate the dataset
    test_data = PeopleFlowsDataset(data_folder, test_df)
    return test_data


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
