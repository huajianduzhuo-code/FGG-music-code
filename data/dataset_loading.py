import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle




class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = torch.Tensor(data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# Custom transform to convert to float16 to avoid type error
class ToFloat16:
    def __call__(self, sample):
        return sample.to(torch.float16)
    

def load_datasets(data_format="separate_melody_accompaniment"):
    '''
    data_format: "combine_melody_accompaniment" or "separate_melody_accompaniment"
    - combine_melody_accompaniment: melody and accompaniment are combined in the same channels
        6 channels, 2 channels for melody and accompaniment, 2 channels for rhythm, 2 channels for null rhythm
    - separate_melody_accompaniment: melody and accompaniment are in separate channels
        8 channels, 2 channels for accompaniment, 2 channels for rhythm, 2 channels for null rhythm, 2 channels for melody
    '''
    # Load data from file
    if data_format == "combine_melody_accompaniment":
        with open('data/train_test_slices/train_slices_combine_melody_accompaniment.pkl', 'rb') as f:
            data = pickle.load(f)
    elif data_format == "separate_melody_accompaniment":
        with open('data/train_test_slices/train_slices_separate_melody_accompaniment.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Invalid data format: {data_format}")
    dataset = CustomDataset(data, transform=ToFloat16())
    return dataset


def create_dataloader(batch_size, dataset):# Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
