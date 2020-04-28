import torch
import torchvision
import numpy as np
from PIL import Image, ImageOps

class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, images):
        return [[self.worker(img) for img in snpt] for snpt in images]

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div
        self.worker = torchvision.transforms.ToTensor()

    def __call__(self, pic):
        images = np.array([[np.array(img) for img in snpt] for snpt in pic])
        images = torch.from_numpy(images).permute(0, 1, 4, 2, 3).contiguous()
        return images.float().div(255) if self.div else images.float()