import os
import time
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from dataset.dataset import *
from dataset.transforms import *

train_dataloader = torch.utils.data.DataLoader(
                    VideoDataset('custom', 'train',
                        transform=torchvision.transforms.Compose([
                            GroupRandomHorizontalFlip(),
                            GroupColorJittering(brightness=0.5, contrast=0.5, saturation=0, hue=0),
                            GroupMultiScaleCrop(256),
                            GroupScale(256),
                            ToTorchFormatTensor(True)])),
                    batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=True)
print(len(train_dataloader))
for data in train_dataloader:
    print(data['frames'].shape)
    print(data['label_idx'])
    print(data['label_name'])