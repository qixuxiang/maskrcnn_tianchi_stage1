# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from PIL import Image
import numpy as np
from numpy import random
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class StretchResize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        min_size = random.choice(self.min_size)
        image = F.resize(image, (min_size, self.max_size))
        #target = target.resize(image.size)
        return image, target


class ToPercentCoords(object):
    def __call__(self, image, target):
        w, h = image.size
        target.bbox[:, 0] /= w
        target.bbox[:, 2] /= w
        target.bbox[:, 1] /= h
        target.bbox[:, 3] /= h

        return image, target

class ToCenterForm(object):
    def __call__(self, image, target):
        cx = (target.bbox[:, 0] + target.bbox[:, 2]) / 2.
        cy = (target.bbox[:, 1] + target.bbox[:, 3]) / 2.
        w = target.bbox[:, 2] - target.bbox[:, 0]
        h = target.bbox[:, 3] - target.bbox[:, 1]
        target.bbox[:, 0] = cx
        target.bbox[:, 1] = cy
        target.bbox[:, 2] = w
        target.bbox[:, 3] = h

        return image, target

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, target):
        if random.randint(2):
            return image, target

        w, h = image.size
        ratio = random.uniform(1, 4)
        left = random.uniform(0, w * ratio - w)
        top = random.uniform(0, h * ratio - h)

        expand_image = np.zeros(
            (int(h * ratio), int(w * ratio), 3),
        )
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + h),
        int(left):int(left + w)] = image
        image = Image.fromarray(expand_image.astype('uint8')).convert('RGB')

        target.bbox[:, :2] += torch.Tensor([int(left), int(top)])
        target.bbox[:, 2:] += torch.Tensor([int(left), int(top)])

        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
