from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from dataset.visdrone import cfg_data
from transformations import DeNormalize


def display_callback(input, prediction, name):
    """
    displays using matplotlib the input image and the prediction
    """
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
    """
    prints the counting predicted
    """
    print(str(name) + ' Count: ' + str(np.round(torch.sum(prediction.squeeze()).item() / cfg_data.LOG_PARA)))


def save_callback(input, prediction, name):
    """
    serialize the prediciton image adding .png to the original file name
    """
    plt.imsave(name + '.png', prediction.squeeze(), cmap='jet')


def video_callback(input, prediction, name):
    """
    show image input and prediction into a cv2 window
    """
    restore = DeNormalize(cfg_data.MEAN, cfg_data.STD)
    GT_SCALE_FACTOR = 2550
    count = str(np.round(torch.sum(prediction).item() / GT_SCALE_FACTOR))
    cm = plt.get_cmap('jet')
    prediction = cm(prediction.squeeze().cpu()).astype(dtype='float32')
    prediction = cv2.cvtColor(prediction, cv2.COLOR_RGBA2BGR)
    input = restore(input).permute(1, 2, 0)
    frame = np.hstack((input, prediction))
    cv2.imshow('Count: ' + str(count), frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        pass


call_dict = {'save_callback': save_callback,
             'count_callback': count_callback,
             'display_callback': display_callback,
             'video_callback': video_callback}
