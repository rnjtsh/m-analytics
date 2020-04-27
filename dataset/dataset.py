import os
import glob
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, dataset_name,
                num_segments=3, modality='RGB',
                transform=None):
        self.image_files_path = os.path.join('data', dataset_name, 'images')
        self.labels_path = os.path.join('data', dataset_name, 'labels')

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass    