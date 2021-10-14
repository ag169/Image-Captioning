from torchvision import transforms as tv_transforms
from PIL import Image
import torchvision.transforms.functional as tfunc
import albumentations.augmentations.functional as afunc
import numpy as np
import random

from utils.gen_ops import round_down, bounding_boxes, max_rot_aclk, max_rot_clk, make_square


# DEFAULT VALUES
DEFAULT_SIZE = 256
DEFAULT_INTERPOLATION_PIL = Image.BILINEAR
DEFAULT_INTERPOLATION = tfunc.InterpolationMode.BILINEAR
DEFAULT_ANGLE = 30


# MAX VALUES
MAX_SIZE = 512


# TRANSFORM FUNCTIONS
def compose(image, label, transforms):
    for transform in transforms:
        image, label = transform(image, label)

    return image, label


def scale(image, label, size, seg_size=None, interp=DEFAULT_INTERPOLATION):
    if seg_size is None:
        seg_size = size
    image = tfunc.resize(image, size, interpolation=interp)
    label = tfunc.resize(label, seg_size, interpolation=interp)
    return image, label


def tensorize(image, label, norm=True):
    image = tfunc.to_tensor(image)
    if norm:
        image = tfunc.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    label = tfunc.to_tensor(label)
    return image, label


def untensorize(image_t, label_t=None, unnorm=True):
    if unnorm:
        image_t = tfunc.normalize(image_t, mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1./0.229, 1./0.224, 1./0.225])

    image_t = image_t.cpu().detach().numpy()

    if label_t is None:
        return image_t

    label_t = label_t.cpu().detach().numpy()

    return image_t, label_t


def square(image, label, size, seg_size=None, stretch=False, interp=DEFAULT_INTERPOLATION_PIL):
    if seg_size is None:
        seg_size = size

    if stretch:
        image = image.resize((size, size), resample=interp)
        label = label.resize((seg_size, seg_size), resample=interp)
    else:
        image = make_square(image, size, 'RGB')
        label = make_square(label, seg_size, 'L')

    return image, label


# TRANSFORM CLASSES

class Scale:
    def __init__(self, size=DEFAULT_SIZE, interpolation=DEFAULT_INTERPOLATION):
        self.size = size
        self.interp = interpolation

    def __call__(self, image, label):
        return scale(image, label, size=self.size, interp=self.interp)


class Square:
    def __init__(self, size=DEFAULT_SIZE, segsize=None, stretch=False, interpolation=DEFAULT_INTERPOLATION):
        self.size = size
        self.segsize = size if segsize is None else segsize
        self.stretch = stretch
        self.interpolation = interpolation

    def __call__(self, image, label):
        return square(image, label, size=self.size, seg_size=self.segsize, stretch=self.stretch,
                      interp=self.interpolation)


class ScaleDown2N:
    def __init__(self, size=DEFAULT_SIZE, segsize=None, n=32, interpolation=DEFAULT_INTERPOLATION):
        size = round_down(size, n)

        self.size = size
        self.segsize = size if segsize is None else round_down(segsize, n)
        self.n = n
        self.interpolation = interpolation

    def __call__(self, image, label):
        w, h = image.size

        if self.segsize == self.size:
            if w < self.size:
                ow_i = self.size
                oh_i = int(h * self.size / w)
                oh_i = round_down(oh_i, self.n)
            else:
                oh_i = self.size
                ow_i = int(w * self.size / h)
                ow_i = round_down(ow_i, self.n)

            return scale(image, label, size=[oh_i, ow_i], interp=self.interpolation)
        else:
            if w < self.size:
                ow_i = self.size
                oh_i = int(h * self.size / w)
                oh_i = round_down(oh_i, self.n)

                ow_l = self.segsize
                oh_l = int(h * self.segsize / w)
                oh_l = round_down(oh_l, self.n)
            else:
                oh_i = self.size
                ow_i = int(w * self.size / h)
                ow_i = round_down(ow_i, self.n)

                oh_l = self.segsize
                ow_l = int(w * self.segsize / h)
                ow_l = round_down(ow_l, self.n)

            return scale(image, label, size=[oh_i, ow_i], seg_size=[oh_l, ow_l], interp=self.interpolation)


class RandomScale:
    def __init__(self, s_min=0.8, s_max=1.2, interpolation=DEFAULT_INTERPOLATION):
        self.min = s_min
        self.max = s_max
        self.interp = interpolation
        self.r1 = random.Random()

    def __call__(self, image, label):
        w, h = image.size

        min_dim = min(w, h)

        min_size = self.min * min_dim
        max_size = self.max * min_dim

        size = self.r1.uniform(min_size, max_size)

        return scale(image, label, size=int(size), interp=self.interp)


class BoundedRandomScaleCrop:
    def __init__(self, s_min=0.8, s_max=1.2, size=DEFAULT_SIZE, segsize=None,
                 n=32, interpolation=DEFAULT_INTERPOLATION):
        self.min = s_min
        self.max = s_max
        size = round_down(size, n)
        self.size = size
        self.segsize = size if segsize is None else round_down(segsize, n)
        self.n = n
        self.interp = interpolation
        self.r1 = random.Random()
        self.r2 = random.Random()
        self.r3 = random.Random()

    def __call__(self, image, label):
        w, h = image.size
        [w1, h1, w2, h2] = bounding_boxes(label)

        wb = w2 - w1
        hb = h2 - h1

        s_max = self.size / max(hb, wb)
        s_min = self.size / min(h, w)

        if s_min <= s_max:
            if s_min < s_max:
                s = self.r1.uniform(max(s_min, self.min), min(s_max, self.max))
            else:
                s = s_min

            new_dims = (int(s * h), int(s * w))
            image, label = scale(image, label, new_dims, interp=self.interp)

            x_min = int(s * w2) - self.size
            x_max = int(s * w1)

            y_min = int(s * h2) - self.size
            y_max = int(s * h1)

            x = self.r2.randint(x_min, x_max)
            y = self.r3.randint(y_min, y_max)

            image = image.crop((x, y, x + self.size, y + self.size))
            label = label.crop((x, y, x + self.size, y + self.size))

            if self.segsize != self.size:
                label = tfunc.resize(label, self.segsize)
        else:
            image, label = square(image, label, size=self.size, seg_size=self.segsize)

        return image, label


class RandomRotate:
    def __init__(self, angle=DEFAULT_ANGLE, p=0.6):
        self.angle = angle
        self.r1 = random.Random()
        self.r2 = random.Random()
        self.prob = p

    def __call__(self, image, label):
        if self.r1.random() < self.prob:
            angle = self.r2.randint(-self.angle, self.angle)
            image = tfunc.rotate(image, angle)
            label = tfunc.rotate(label, angle)

        return image, label


class BoundedRandomRotate:
    def __init__(self, angle=DEFAULT_ANGLE, p=0.6):
        self.angle = angle
        self.r1 = random.Random()
        self.r2 = random.Random()
        self.prob = p

    def __call__(self, image, label):
        if self.r1.random() < self.prob:
            c = bounding_boxes(label)
            deg1 = max_rot_clk(label, bb=c, max_deg=self.angle, op_int=True, margin=0)
            deg2 = max_rot_aclk(label, bb=c, max_deg=self.angle, op_int=True, margin=0)

            angle = self.r2.randint(-deg2, deg1)
            image = tfunc.rotate(image, angle)
            label = tfunc.rotate(label, angle)

        return image, label


class RandomCrop:
    def __init__(self, size=DEFAULT_SIZE, segsize=None):
        self.size = size
        self.segsize = size if segsize is None else segsize
        self.r1 = random.Random()
        self.r2 = random.Random()

    def __call__(self, image, label):
        w, h = image.size

        min_dim = min(w, h)

        if min_dim < self.size:
            image, label = scale(image, label, self.size)
            w, h = image.size

        x = self.r1.randint(0, w - self.size) if w > self.size else 0
        y = self.r2.randint(0, h - self.size) if h > self.size else 0

        image = tfunc.crop(image, x, y, self.size, self.size)
        label = tfunc.crop(label, x, y, self.size, self.size)

        if self.segsize != self.size:
            label = tfunc.resize(label, self.segsize)

        return image, label


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = tv_transforms.ColorJitter(brightness=brightness,
                                                      contrast=contrast,
                                                      saturation=saturation,
                                                      hue=hue)

    def __call__(self, image, label):
        return self.color_jitter(image), label


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.r1 = random.Random()
        self.p = p

    def __call__(self, image, label):
        if self.r1.random() < self.p:
            return tfunc.hflip(image), tfunc.hflip(label)

        return image, label


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.r1 = random.Random()
        self.p = p

    def __call__(self, image, label):
        if self.r1.random() < self.p:
            return tfunc.vflip(image), tfunc.vflip(label)

        return image, label


class RandomNoise:
    def __init__(self, p=0.1):
        self.r1 = random.Random()
        self.p = p

    def __call__(self, image, label):
        if self.r1.random() < self.p:
            return Image.fromarray(afunc.iso_noise(np.array(image))), label
        return image, label

