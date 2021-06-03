import torch
import os
from PIL import Image as pil
import numpy as np
import mimetypes
import cv2


class FilesDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def set_transforms(self, transforms):
        self.transforms = transforms

    def __getitem__(self, item):
        with pil.open(self.files[item]) as img:
            data = np.array(img)
        if self.transforms:
            data = self.transforms(data)
        return data, self.files[item]

    def __len__(self):
        return len(self.files)


class FolderDataset(FilesDataset):
    def __init__(self, folder, transforms=None):
        files = [os.path.join(folder, self.files[file]) for file in os.listdir(folder)]
        super().__init__(files, transforms)


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_file):
        self.video = cv2.VideoCapture(video_file)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = 0

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        ret, data = self.video.read()

        if self.transforms:
            data = self.transforms(data)
        self.frame_count += 1
        return data, self.frame_count - 1


def make_dataset(input):
    if issubclass(type(input), str):
        if os.path.isdir(input):
            return FolderDataset(input)
        elif os.path.isfile(input):
            mimetypes.init()
            mimestart = mimetypes.guess_type(input)[0]

            if mimestart != None:
                mimestart = mimestart.split('/')[0]

                if mimestart == 'image':
                    return FilesDataset([input])
                if mimestart == 'video':
                    return VideoDataset(input)
    elif issubclass(type(input), list):
        return FilesDataset(input)
    raise Exception('Input type not recognized!')
