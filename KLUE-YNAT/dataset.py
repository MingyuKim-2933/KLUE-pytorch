import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, text, label):

        self.x = text
        self.y = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.x[idx])
        y = torch.LongTensor(self.y[idx])

        return x, y