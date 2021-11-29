import torch
import torchvision.transforms.functional as tfunc
from PIL import Image
from utils.gen_ops import make_square
import random
from albumentations.augmentations import ISONoise as alb_ISONoise
import numpy as np


DEFAULT_SIZE = 256
DEFAULT_INTERPOLATION_PIL = Image.BILINEAR
DEFAULT_INTERPOLATION = tfunc.InterpolationMode.BILINEAR
DEFAULT_ANGLE = 30


def square(image, size, stretch=False, interp=DEFAULT_INTERPOLATION_PIL):
    if stretch:
        image = image.resize((size, size), resample=interp)
    else:
        image = make_square(image, size, 'RGB')

    return image


class Square(torch.nn.Module):
    def __init__(self, size=DEFAULT_SIZE, stretch=False, interpolation=DEFAULT_INTERPOLATION):
        super(Square, self).__init__()
        self.size = size
        self.stretch = stretch
        self.interpolation = interpolation

    def __call__(self, image):
        return square(image, size=self.size, stretch=self.stretch, interp=self.interpolation)


class RandomScale(torch.nn.Module):
    def __init__(self, s_min=0.8, s_max=1.2, interpolation=DEFAULT_INTERPOLATION):
        super(RandomScale, self).__init__()
        self.min = s_min
        self.max = s_max
        self.interp = interpolation
        self.r1 = random.Random()

    def __call__(self, image):
        w, h = image.size

        min_dim = min(w, h)

        min_size = self.min * min_dim
        max_size = self.max * min_dim

        size = int(self.r1.uniform(min_size, max_size))

        return tfunc.resize(image, size=size, interpolation=self.interp)


class RandomCrop(torch.nn.Module):
    def __init__(self, size=DEFAULT_SIZE):
        super(RandomCrop, self).__init__()
        self.size = size
        self.r1 = random.Random()
        self.r2 = random.Random()
        self.sq = Square(size=size, stretch=False)

    def __call__(self, image):
        w, h = image.size

        min_dim = min(w, h)

        if min_dim <= self.size:
            image = self.sq(image)
            # w, h = image.size
        else:
            x = self.r1.randint(0, w - self.size) if w > self.size else 0
            y = self.r2.randint(0, h - self.size) if h > self.size else 0

            image = tfunc.crop(image, x, y, self.size, self.size)

        return image


class ISONoise(torch.nn.Module):
    def __init__(self, p=0.5):
        super(ISONoise, self).__init__()
        self.p = p
        self.r1 = random.Random()
        self.iso_noise = alb_ISONoise(always_apply=True)

    def __call__(self, image):
        r = self.r1.random()

        if r > self.p:
            return image

        image_np = np.array(image)
        augmented = self.iso_noise(image=image_np)['image']

        return Image.fromarray(augmented)
