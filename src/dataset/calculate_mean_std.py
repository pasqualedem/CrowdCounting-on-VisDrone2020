import multiprocessing

import numpy as np
from tqdm import tqdm
import os
from PIL import Image as pil
from multiprocessing import Pool

WORKERS = multiprocessing.cpu_count()


def mean_std_img(path):
    img = pil.open(path)
    img = np.array(img)
    means = np.mean((img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2]), axis=0)
    stds = np.std((img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2]), axis=0)
    return means, stds


def calculate():
    """
    Calculate the mean and the standard deviation of a dataset
    """
    folder = 'dataset/VisDrone2020-CC/raw/train'
    folders = os.listdir(folder)
    raw_path, subset = os.path.split(folder)
    data_superpath, _ = os.path.split(raw_path)
    dataset = []
    for cur_folder in folders:
        files = os.listdir(os.path.join(folder, cur_folder))
        for file in files:
            dataset.append(os.path.join(folder, cur_folder, file))
    length = len(dataset)
    means = 0
    stds = 0

    try:
        with Pool(WORKERS) as p:
            result = np.array(list(tqdm(p.imap(mean_std_img, dataset), total=length)))
        means_stds = result.sum(axis=0)
        means, stds = means_stds[0], means_stds[1]

    finally:
        print(means / length)
        print(stds / length)
        return means/length, stds/length


if __name__ == '__main__':
    calculate()
