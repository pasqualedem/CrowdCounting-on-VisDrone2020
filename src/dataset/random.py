import torch


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, dim, n):
        self.dim = dim
        self.n = n

    def __getitem__(self, item):
        return torch.rand(self.dim).cuda()

    def __len__(self):
        return self.n

    def shape(self):
        return self.dim