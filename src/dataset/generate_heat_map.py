import os

import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import re

GAMMA = 3
FILENAME_LEN = 5


def generate_heatmap(df, img_size, wished_heatmap_size):
    """
    Generate a dictionary of heatmaps

    @param df: dataframe of columns [frame, x, y]
    @param img_size: original size of the image
    @param wished_heatmap_size: parameter for resizing the heatmap
    @return:
    """
    heatmaps = {}
    frames = np.unique(df['frame'].values)
    img_size = np.array(img_size)
    wished_heatmap_size = np.array(wished_heatmap_size)
    ratio = (img_size / wished_heatmap_size)
    for frame in frames:
        heads = np.rint(df[df['frame'] == frame][['x', 'y']].values / ratio).astype('int64')
        heatmap = np.zeros(wished_heatmap_size)
        heatmap[np.clip(heads[:, 1], 0, wished_heatmap_size[0]) - 1,
                np.clip(heads[:, 0] - 1, 0, wished_heatmap_size[1]) - 1] = 1
        heatmap = gaussian_filter(heatmap, GAMMA / ratio)
        heatmaps[frame] = heatmap
    return heatmaps


def make_ground_truth(folder, img_folder, name_rule, img_rule, dataframe_fun, size=None):
    """
    Generate the ground truth h5 files

    :param folder: The folder where the annotations are stored
    :param img_folder: The folder where the images are stored
    :param name_rule: rule for including annotations files based on their name
    :param img_rule: rule for obtaining the img name file from the folder and annotation name file
    :param dataframe_fun: function that returns a [frame, x, y] dataframe given the annotation file name
    :param size: wished size of heatmap
    """
    gt_files = os.listdir(folder)
    gt_files = list(filter(name_rule, gt_files))

    for gt in tqdm(gt_files):
        fname, ext = gt.split('.')
        seq_folder = img_rule(img_folder, fname)
        img_size = plt.imread(
            os.path.join(seq_folder,
                         list(filter(lambda x: '.jpg' in x, os.listdir(seq_folder)))[0]
                         )).shape[:2]

        if size is None:
            size = img_size
        df = dataframe_fun(os.path.join(folder, gt))
        heatmaps = generate_heatmap(df, img_size, size)
        for heatmap in heatmaps:
            hf = h5py.File(
                os.path.join(
                    seq_folder,
                    (str(heatmap).zfill(FILENAME_LEN) + re.sub(', |\(|\)', '_', str(size)) + '.h5')), 'w')
            hf.create_dataset('density', data=heatmaps[heatmap])
            hf.close()


def dataframe_load_test(filename):
    """
    Load the dataframe for the test set csv format of VisDrone
    @param filename: csv path
    @return: dataframe of columns [frame, x, y]
    """
    df = pd.read_csv(filename, header=None)
    df.columns = ['frame', 'head_id', 'x', 'y', 'width', 'height', 'out', 'occl', 'undefinied', 'undefinied']
    df['x'] = df['x'] + df['width'] // 2
    df['y'] = df['y'] + df['height'] // 2

    df = df[(df['frame'] % 10) == 1]
    df['frame'] = df['frame'] // 10 + 1
    return df[['frame', 'x', 'y']]


def dataframe_load_train(filename):
    """
    Load the dataframe for the training set csv format of VisDrone
    @param filename: csv path
    @return: dataframe of columns [frame, x, y]
    """
    df = pd.read_csv(filename, header=None)
    df.columns = ['frame', 'x', 'y']
    return df


if __name__ == '__main__':
    train_rule = lambda x: '.txt' in x and 'clean' not in x
    test_rule = lambda x: '_clean.txt' in x

    img_train_rule = lambda x, y: os.path.join(x, y)
    img_test_rule = lambda x, y: os.path.join(x, re.sub('\_clean$', '', y))

    size = (540, 960)

    train = [train_rule, img_train_rule, dataframe_load_train, size]
    test = [test_rule, img_test_rule, dataframe_load_test, size]

    make_ground_truth('../../dataset/VisDrone2020-CC/annotations',
                      '../../dataset/VisDrone2020-CC/train',
                      *train)
