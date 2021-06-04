from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from dataset.visdrone import cfg_data


def display_callback(input, prediction, name):
    GT_SCALE_FACTOR = 2550
    img = Image.open(name)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 8))
    fig.suptitle("Count of " + name + " : " + str(np.round(torch.sum(prediction).item() / GT_SCALE_FACTOR)))
    ax1.axis('off')
    ax1.imshow(img)
    ax2.axis('off')
    ax2.imshow(prediction.squeeze().numpy(), cmap='jet')
    plt.show()
    print()


def count_callback(input, prediction, name):
    print(str(name) + ' Count: ' + str(np.round(torch.sum(prediction.squeeze()).item() / cfg_data.LOG_PARA)))


def save_callback(input, prediction, name):
    plt.imsave(name + '.png', prediction.squeeze(), cmap='jet')


call_dict = {'save_callback': save_callback, 'count_callback': count_callback, 'display_callback': display_callback}
