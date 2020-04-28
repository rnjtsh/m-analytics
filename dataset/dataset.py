import os
import glob
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, dataset_name, split_type,
                modality='RGB', transform=None):
        self.dataset_name = dataset_name
        self.split_type = split_type
        self.image_files = os.path.join('data', dataset_name, split_type)
        self.class_dict = self._get_classes()
        self.modality = modality
        self.transform = transform
        self.video_list = self._get_video_list()

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        _video = self.video_list[idx]
        _label = os.path.split(os.path.dirname(_video))[-1]
        _snippets = glob.glob(_video + '/*')
        images = []
        for snippet in _snippets:
            _image_files = glob.glob(snippet + '/*')
            _images = []
            for image_file in _image_files:
                _images.append(Image.open(image_file).convert('RGB'))
            images.append(_images)
        return {'frames': self.transform(images), 'label_idx': self.class_dict[_label], 'label_name': _label}

    def _get_classes(self):
        class_files = os.path.join('data', self.dataset_name, 'class_list.txt')
        with open(class_files, 'r') as f:
            _classes = f.read().splitlines()
            _classes = sorted(_classes)
        return dict(zip(_classes, range(len(_classes))))

    def _get_video_list(self):
        _class_folders = glob.glob(self.image_files + '/*')
        _video_list = []
        for class_folder in _class_folders:
            _video_list.extend(glob.glob(class_folder + '/*'))
        return _video_list