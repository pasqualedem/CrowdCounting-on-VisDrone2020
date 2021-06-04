import numpy as np
from tqdm import tqdm
from dataset.visdrone import VisDroneDataset, make_dataframe


def calculate():
    """
    Calculate the mean and the standard deviation of a dataset
    """
    df = make_dataframe('../dataset/VisDrone2020-CC/sequences')
    ds = VisDroneDataset(df, train=False)
    length = len(ds)
    means = 0
    stds = 0
    i = 0
    try:
        for img, den in tqdm(ds):
            means += np.mean((img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2]), axis=0)
            stds += np.std((img / 255).reshape(img.shape[0] * img.shape[1], img.shape[2]), axis=0)
            i += 1
    finally:
        print(means / length)
        print(stds / length)
        print(i)


if __name__ == '__main__':
    calculate()
