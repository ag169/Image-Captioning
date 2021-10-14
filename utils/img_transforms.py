import torch
import torchvision.transforms.functional as tfunc
from PIL import Image
from utils.gen_ops import make_square


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


