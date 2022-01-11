import numpy as np
import pandas as pd
import os
import torch
import torchvision
from PIL import Image as pil
import h5py
import re
from config import cfg
from easydict import EasyDict
import sklearn.model_selection
import transformations as trans

cfg_data = EasyDict()


cfg_data.DATA_PATH = '../dataset/VisDrone2020-CC'
cfg_data.SIZE = (1080, 1920)
cfg_data.FILE_EXTENSION = '.jpg'
cfg_data.GT_FILE_EXTENSION = '.h5'
cfg_data.LOG_PARA = 2550.0

cfg_data.GAMMA_CORRECTION = False
cfg_data.BETA_ALPHA = 4.2
cfg_data.BETA_BETA = 2.4

cfg_data.MEAN = [0.43476477, 0.44504763, 0.43252817]
cfg_data.STD = [0.20490805, 0.19712372, 0.20312176]
cfg_data.DATA_SUBFOLDER = 'raw'


class VisDroneDataset(torch.utils.data.Dataset):
    """
    Dataset subclass for the VisDrone dataset
    """
    def __init__(self, dataframe, train=True, img_transform=True, gt_transform=True):
        """
        Initialize VisDroneDataset object
        @param dataframe: dataframe of columns [id, filename, gt_filename] for loading images and ground truth
        @param train: boolean that specify if dataset is loaded for train (it applies the Random Horizontal Flip)
        @param img_transform: boolean that specify if scaling, normalizing and resizing data
        @param gt_transform: boolean that specify if multiply the GT to the LOG_PARA constant
        """
        self.dataframe = dataframe
        self.train_transforms = None
        self.img_transform = None
        self.gt_transform = None
        if train:
            self.train_transforms = trans.RandomHorizontallyFlip()

        if img_transform:
            # Initialize data transforms
            trans_list = [torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean=cfg_data.MEAN,
                                                           std=cfg_data.STD),
                          torchvision.transforms.Resize(cfg_data.SIZE)
                          ]
            if cfg_data.GAMMA_CORRECTION:
                trans_list.insert(1, trans.RandomGammaCorrection(cfg_data.BETA_ALPHA, cfg_data.BETA_BETA))
            self.img_transform = torchvision.transforms.Compose(trans_list)  # normalize to (-1, 1)

        if gt_transform:
            self.gt_transform = torchvision.transforms.Compose([
                trans.Scale(cfg_data.LOG_PARA)
            ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, i):
        """
        Retrieve, load and preprocess an item of the dataset and its ground truth

        @param i: the id in the dataframe of the item
        @return: data and ground truth tensors
        """
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


def make_dataframe(folder, size):
    """
    Given a folder requiring to have subfolders each one containing the the frames
    and a parallel folder containing the .h5 ground truths,
    builds a dataframe tracking all the dataset files

    @param folder: the path folder from where build the dataframe
    @param size: size of the heatmaps required to find them
    @return: a DataFrame with columns (example folder, example idx, filename, gt filename)
    """
    folders = os.listdir(folder)
    raw_path, subset = os.path.split(folder)
    data_superpath, _ = os.path.split(raw_path)
    h5_folder = os.path.join(data_superpath, 'processed', 'heatmaps' + re.sub(', |\(|\)', '_', str(size)), subset)
    dataset = []
    for cur_folder in folders:
        files = os.listdir(os.path.join(folder, cur_folder))
        for file in files:
            if cfg_data.FILE_EXTENSION in file:
                idx, ext = file.split('.')
                gt = os.path.join(h5_folder, cur_folder, idx + cfg_data.GT_FILE_EXTENSION)
                dataset.append([idx, os.path.join(folder, cur_folder, file), gt])
    return pd.DataFrame(dataset, columns=['id', 'filename', 'gt_filename'])


def load_test():
    """
    Create a VisDroneDataset object in test mode
    @return: the visdrone testset
    """
    df = make_dataframe(os.path.join(cfg_data.DATA_PATH, cfg_data.DATA_SUBFOLDER, 'test'), cfg_data.SIZE)
    ds = VisDroneDataset(df, train=cfg.TRAIN, gt_transform=cfg.GT_TRANSFORM)
    return ds


def load_train_val():
    """
    Create a train and validation DataLoader from the specified folder
    config values are used (VAL_SIZE, VAL_BATCH_SIZE, N_WORKERS)

    @return: the train and validation DataLoader
    """
    # train_df = make_dataframe('../dataset/VisDrone2020-CC/train')
    # valid_df = make_dataframe('../dataset/VisDrone2020-CC/val')

    df = make_dataframe(os.path.join(cfg_data.DATA_PATH, cfg_data.DATA_SUBFOLDER, 'train'), cfg_data.SIZE)
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


# def generate_validation():
#     import os
#     import numpy as np
#     import shutil
#     np.random.seed(cfg.SEED)
#     source = "../../dataset/VisDrone2020-CC/train"
#     target = "../val"
#     os.chdir(source)
#     l = os.listdir()
#     val = np.random.choice(l, 16, replace=False)  # 16 is the 0.20 of the training set
#     for file in val:
#         shutil.move(os.path.join('./', file), os.path.join(target, file))
