import torch


class RandomDataset(torch.utils.data.Dataset):
    """
    Generate a random dataset given the dimension and number of elements
    """
    def __init__(self, dim, n):
        self.dim = dim
        self.n = n

    def __getitem__(self, item):
        return torch.rand(self.dim)

    def __len__(self):
        return self.n

    def shape(self):
        return self.dim