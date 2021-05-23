import os

import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
GAMMA = 3
FILENAME_LEN = 5


def generate_heatmap(filename, dim):
    df = pd.read_csv(filename)
    df.columns = ['frame', 'x', 'y']
    heatmaps = {}
    frames = np.unique(df['frame'].values)
    for frame in frames:
        heads = df[df['frame'] == frame][['x', 'y']].values
        heatmap = np.zeros(dim)
        heatmap[heads[:, 1] - 1, heads[:, 0] - 1] = 1
        heatmap = gaussian_filter(heatmap, GAMMA)
        heatmaps[frame] = heatmap
    return heatmaps


def make_ground_truth(folder, img_folder):
    gt_files = os.listdir(folder)
    for gt in tqdm(gt_files):
        fname, ext = gt.split('.')
        seq_folder = os.path.join(img_folder, fname)
        dim = plt.imread(os.path.join(seq_folder, '00001.jpg')).shape[:2]
        heatmaps = generate_heatmap(os.path.join(folder, gt), dim)
        for heatmap in heatmaps:
            plt.imsave(arr=heatmaps[heatmap],
                       fname=os.path.join(seq_folder, (str(heatmap).zfill(FILENAME_LEN) + '.png')),
                       cmap='viridis',
                       format='png')


if __name__ == '__main__':
    make_ground_truth('../dataset/VisDrone2020-CC/annotations', '../dataset/VisDrone2020-CC/sequences')