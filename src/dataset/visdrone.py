import numpy as np
import pandas as pd
import os
import torch
import torchvision
from PIL import Image as pil
import h5py
from config import cfg
from easydict import EasyDict
import sklearn.model_selection
import transformations as trans

cfg_data = EasyDict()

cfg_data.SIZE = ()
cfg_data.FILE_EXTENSION = '.jpg'
cfg_data.GT_FILE_EXTENSION = '.h5'
cfg_data.LOG_PARA = 2550.0

MEAN = [0.43476477, 0.44504763, 0.43252817]
STD = [0.20490805, 0.19712372, 0.20312176]


class VisDroneDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, train=True, img_transform=True, gt_transform=True):
        self.dataframe = dataframe
        self.train_transforms = None
        self.img_transform = None
        self.gt_transform = None
        if train:
            self.train_transforms = trans.RandomHorizontallyFlip()

        if img_transform:
            # Initialize data transforms
            self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                 torchvision.transforms.Normalize(mean=MEAN,
                                                                                                  std=STD),
                                                                 ])  # normalize to (-1, 1)
        if gt_transform:
            self.gt_transform = torchvision.transforms.Compose([
                trans.Scale(cfg_data.LOG_PARA)
            ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        # Obtain the filename and target
        filename = self.dataframe.loc[i]['filename']
        target_filename = self.dataframe.loc[i]['gt_filename']

        # Load the img and the ground truth
        with pil.open(filename) as img:
            data = np.array(img)
        hf = h5py.File(target_filename, 'r')
        target = np.array(hf.get('density'))
        hf.close()
        if self.train_transforms:
            data, target = self.train_transforms(data, target)

        if self.img_transform:
            data = self.img_transform(data)

        if self.gt_transform:
            target = self.gt_transform(target)

        return data, target

    def get_targets(self):
        return self.targets


def make_dataframe(folder):
    # Return a DataFrame with columns (example folder, example idx, filename, gt filename)
    folders = os.listdir(folder)
    dataset = []
    for cur_folder in folders:
        files = os.listdir(os.path.join(folder, cur_folder))
        for file in files:
            if cfg_data.FILE_EXTENSION in file:
                idx, ext = file.split('.')
                gt = os.path.join(folder, cur_folder, idx + cfg_data.GT_FILE_EXTENSION)
                dataset.append([idx, os.path.join(folder, cur_folder, file), gt])
    return pd.DataFrame(dataset, columns=['id', 'filename', 'gt_filename'])


def load_test():
    df = make_dataframe('../dataset/VisDrone2020-CC/test')
    ds = VisDroneDataset(df, train=False)
    return ds


def load_train_val():
    df = make_dataframe('../dataset/VisDrone2020-CC/sequences')

    # Split the dataframe in train and validation
    train_df, valid_df = sklearn.model_selection.train_test_split(
        df, test_size=cfg.VAL_SIZE, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    train_set = VisDroneDataset(train_df)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=cfg.N_WORKERS, shuffle=True)

    val_set = VisDroneDataset(valid_df)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.VAL_BATCH_SIZE, num_workers=cfg.N_WORKERS, shuffle=True)

    return train_loader, val_loader


def load_test_dataset(data_folder, dataclass, make_dataframe_fun):
    # Load the test dataframe
    test_df = make_dataframe_fun(data_folder)

    # Instantiate the dataset
    test_data = dataclass(data_folder, test_df)
    return test_data