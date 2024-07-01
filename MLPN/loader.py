from __future__ import print_function, division

import os
import math
from shutil import copyfile
from os.path import join as ospj
from PIL import Image
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.transforms import InterpolationMode

import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
import imgaug.augmenters as iaa

from LPN.folder import ImageFolder

###############################################





###########################################################

import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from os.path import join as ospj
from PIL import Image
import os
import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from os.path import join as ospj
from PIL import Image
import numpy as np
import os
import torch
from collections import defaultdict
import random


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


# getting one image of a folder.
def make_dataset_one(dir, class_to_idx, extensions, reverse=False):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            index = 0
            for fname in sorted(fnames, reverse=reverse):
                index += 1
                if has_file_allowed_extension(fname, extensions) and index == 36:
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    break

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
        except Exception:
            print(111)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class customData(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, rotate=0, pad=0):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.rotate = rotate
        self.pad = pad

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        img = transforms.functional.rotate(img, self.rotate)
        if self.pad > 0:
            img = transforms.functional.resize(img, (256, 256), interpolation=3)
            img = transforms.functional.pad(img, (self.pad, 0, 0, 0))
            img = transforms.functional.five_crop(img, (256, 256))[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class customData_one(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, rotate=0, pad=0,
                 reverse=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_one(root, class_to_idx, IMG_EXTENSIONS, reverse)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.rotate = rotate
        self.pad = pad

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        img = transforms.functional.rotate(img, self.rotate)
        if self.pad > 0:
            img = transforms.functional.resize(img, (256, 256), interpolation=3)
            img = transforms.functional.pad(img, (self.pad, 0, 0, 0))
            img = transforms.functional.five_crop(img, (256, 256))[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


# ----------------------------------------------------------------------------#
def make_dataset_160k_sat(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, os.path.splitext(fname)[0])
                images.append(item)
    return images


class CustomData160k_sat(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, rotate=0, pad=0):
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.webp']
        imgs = make_dataset_160k_sat(root, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.rotate = rotate
        self.pad = pad

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        img = transforms.functional.rotate(img, self.rotate)
        if self.pad > 0:
            img = transforms.functional.resize(img, (256, 256), interpolation=3)
            img = transforms.functional.pad(img, (self.pad, 0, 0, 0))
            img = transforms.functional.five_crop(img, (256, 256))[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def make_dataset_160k_drone(dir, extensions, query_name):
    images = []
    dir = os.path.expanduser(dir)
    with open(query_name, 'r') as f:
        fnames = f.readlines()
        for fname in fnames:
            fname = fname.strip()
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(dir, fname)
                item = (path, os.path.splitext(fname)[0])
                images.append(item)
    return images


class CustomData160k_drone(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, query_name=None, rotate=0,
                 pad=0):
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.webp']
        imgs = make_dataset_160k_drone(root, IMG_EXTENSIONS, query_name)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.rotate = rotate
        self.pad = pad

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        img = transforms.functional.rotate(img, self.rotate)
        if self.pad > 0:
            img = transforms.functional.resize(img, (256, 256), interpolation=3)
            img = transforms.functional.pad(img, (self.pad, 0, 0, 0))
            img = transforms.functional.five_crop(img, (256, 256))[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


# IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.webp']
# query_drone = make_dataset_160k_drone('/home/zhangqiang/University-Release/test/query_drone_160k', IMG_EXTENSIONS, 'query_drone_name.txt')
# print(query_drone[0])

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


# getting one image of a folder.
def make_dataset_one(dir, class_to_idx, extensions, reverse=False):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            index = 0
            for fname in sorted(fnames, reverse=reverse):
                index += 1
                if has_file_allowed_extension(fname, extensions) and index == 36:
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    break

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        try:
            img = Image.open(f)
        except Exception:
            print(111)

        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class customData(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, rotate=0, pad=0):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.rotate = rotate
        self.pad = pad

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        img = transforms.functional.rotate(img, self.rotate)
        if self.pad > 0:
            img = transforms.functional.resize(img, (256, 256), interpolation=3)
            img = transforms.functional.pad(img, (self.pad, 0, 0, 0))
            img = transforms.functional.five_crop(img, (256, 256))[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class customData_one(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, rotate=0, pad=0,
                 reverse=False):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_one(root, class_to_idx, IMG_EXTENSIONS, reverse)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.rotate = rotate
        self.pad = pad

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        img = transforms.functional.rotate(img, self.rotate)
        if self.pad > 0:
            img = transforms.functional.resize(img, (256, 256), interpolation=3)
            img = transforms.functional.pad(img, (self.pad, 0, 0, 0))
            img = transforms.functional.five_crop(img, (256, 256))[0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def make_pair_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, target, class_to_idx[target])
                    images.append(item)
    return images


class SatData(Data.Dataset):
    def __init__(self, root, transform=None, d_transform=None, loader=default_loader, view='/drone/',
                 dataset='university'):
        if dataset == 'cvact':
            sat_root = root + '/satview_polish/'
        else:
            sat_root = root + '/satellite/'
        classes, class_to_idx = find_classes(sat_root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(sat_root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.d_transform = d_transform
        self.loader = loader
        self.view = view

    def _get_pair_sample(self, root, _cls):
        img_path = []
        folder_root = root + str(_cls)
        assert os.path.isdir(folder_root), 'no pair drone image'
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        return result_path

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, _cls, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        d_root = self.root + self.view
        d_path = self._get_pair_sample(d_root, _cls)
        d_img = self.loader(d_path)
        if self.d_transform is not None:
            d_img = self.d_transform(d_img)
        return img, d_img, target

    def __len__(self):
        return len(self.imgs)


class DroneData(Data.Dataset):
    def __init__(self, root, transform=None, s_transform=None, loader=default_loader, view='/drone/',
                 dataset='university'):
        drone_root = root + view
        classes, class_to_idx = find_classes(drone_root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(drone_root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.s_transform = s_transform
        self.loader = loader
        self.dataset = dataset

    def _get_pair_sample(self, root, _cls):
        img_path = []
        folder_root = root + str(_cls)
        assert os.path.isdir(folder_root), 'no pair sat image'
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        return result_path

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, _cls, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.dataset == 'cvact':
            s_root = self.root + '/satview_polish/'
        else:
            s_root = self.root + '/satellite/'
        s_path = self._get_pair_sample(s_root, _cls)
        s_img = self.loader(s_path)
        if self.s_transform is not None:
            s_img = self.s_transform(s_img)
        return img, s_img, target

    def __len__(self):
        return len(self.imgs)


class AugSatData(Data.Dataset):
    def __init__(self, root, transform=None, d_transform=None, loader=default_loader, view='/drone/',
                 dataset='university'):
        if dataset == 'cvact':
            sat_root = root + '/satview_polish/'
        else:
            sat_root = root + '/satellite/'
        classes, class_to_idx = find_classes(sat_root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(sat_root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.d_transform = d_transform
        self.loader = loader
        self.view = view

    def _get_pair_sample(self, root, _cls):
        img_path = []
        folder_root = root + str(_cls)
        assert os.path.isdir(folder_root), 'no pair drone image'
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        return result_path

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, _cls, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
        d_root = self.root + self.view
        d_path = self._get_pair_sample(d_root, _cls)
        d_img = self.loader(d_path)
        if self.d_transform is not None:
            d_img1 = self.d_transform[0](d_img)
            d_img2 = self.d_transform[1](d_img)
        return img1, img2, d_img1, d_img2, target

    def __len__(self):
        return len(self.imgs)


class AugDroneData(Data.Dataset):
    def __init__(self, root, transform=None, s_transform=None, loader=default_loader, view='/drone/',
                 dataset='university'):
        drone_root = root + view
        classes, class_to_idx = find_classes(drone_root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(drone_root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.s_transform = s_transform
        self.loader = loader
        self.dataset = dataset

    def _get_pair_sample(self, root, _cls):
        img_path = []
        folder_root = root + str(_cls)
        assert os.path.isdir(folder_root), 'no pair sat image'
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        return result_path

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, _cls, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
        if self.dataset == 'cvact':
            s_root = self.root + '/satview_polish/'
        else:
            s_root = self.root + '/satellite/'
        s_path = self._get_pair_sample(s_root, _cls)
        s_img = self.loader(s_path)
        if self.s_transform is not None:
            s_img1 = self.s_transform[0](s_img)
            s_img2 = self.s_transform[1](s_img)
        return img1, img2, s_img1, s_img2, target

    def __len__(self):
        return len(self.imgs)


def make_dataset_selectID(dir, class_to_idx, extensions):
    images = defaultdict(list)
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images[class_to_idx[target]].append(item)
    return images


class ImageFolder_selectID(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset_selectID(root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs.keys()) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = random.choice(self.imgs[index])
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        placeholder = torch.zeros_like(img)
        return img, placeholder, target

    def __len__(self):
        return len(self.imgs)


class ImageFolder_expandID(Data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        classes, class_to_idx = find_classes(root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        imgs = imgs * 3
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        placeholder = torch.zeros_like(img)
        return img, placeholder, target

    def __len__(self):
        return len(self.imgs)

os.environ['TORCH_HOME']='./'

environments = {'normal': iaa.Sequential([iaa.Noop()]),
                'dark' : iaa.Sequential([
                                        # iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31),
                                        iaa.MultiplyAndAddToBrightness(mul=0.4, add=-15, seed=1991)]),
                'fog'  : iaa.Sequential([iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
                                        alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9, density_multiplier=0.5, seed=35)]),
                'rain' : iaa.Sequential([iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
                                        iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
                                        iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
                                        iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
                                        iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95)]),
                'snow' : iaa.Sequential([iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
                                        iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
                                        iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
                                        iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
                                        iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96)]),
                'fog_rain' : iaa.Sequential([iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
                                            alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9, density_multiplier=0.5, seed=35),
                                            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
                                            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)]),
                'fog_snow' : iaa.Sequential([iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
                                            alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9, density_multiplier=0.5, seed=35),
                                            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
                                            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)]),
                'rain_snow' : iaa.Sequential([iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
                                            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
                                            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
                                            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
                                            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74)]),
                'light': iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992)]),
                'wind' : iaa.Sequential([iaa.MotionBlur(15, seed=17)])
                }

class WeatherTransform:
    def __init__(self, aug='normal') -> None:
        self.transform = environments[aug]

    def __call__(self, img):
        img = np.array(img) # input is PIL
        img = self.transform(image=img)
        img = Image.fromarray(img)
        return img
    

class MyDroneData_train(Data.Dataset):
    def __init__(self, root, transform=None, s_transform=None, loader=default_loader, view='/drone/',
                 style='normal',
                 dataset='university',
                 h=256, w=256, pad=10, erasing_p=0, color_jitter=True, DA=True):
        drone_root = root + view
        classes, class_to_idx = find_classes(drone_root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(drone_root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        ###########################
        self.envir_list = [i for i in environments]
        self.style_list = self.envir_list + ['mixed']
        assert style in self.style_list, f"style must be one of {self.style_list}"
        # assert stage in ['train', 'test'], f"style must be one of {['train', 'test']}"
        self.style = style 
        ###################################

        transform_train_list = [
            # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
            transforms.Pad(pad, padding_mode='edge'),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        transform_satellite_list = [
            transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
            transforms.Pad(pad, padding_mode='edge'),
            transforms.RandomAffine(90),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        transform_val_list = [
            transforms.Resize(size=(h, w), interpolation=3),  # Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        if color_jitter:
            transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                    hue=0)] + transform_train_list
            transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                        hue=0)] + transform_satellite_list
                                                        

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform_train_list
        self.s_transform = transform_satellite_list
        self.loader = loader
        self.dataset = dataset

    def _get_pair_sample(self, root, _cls):
        img_path = []
        folder_root = root + str(_cls)
        assert os.path.isdir(folder_root), 'no pair sat image'
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        return result_path

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        # set weather 
        if self.style=='mixed':
            weather = np.random.choice(self.envir_list)
        else:
            weather = self.style

        transforms_d = self.transform
        if weather != 'normal':    
            # transforms_d.insert(2, WeatherTransform(aug=weather))
            transforms_d = [None] + transforms_d
            transforms_d[0] = WeatherTransform(aug=weather)

        # load img 1
        transform_ = transforms.Compose(transforms_d)
        path, _cls, target = self.imgs[index]
        img = self.loader(path)
        img = transform_(img)

        # load img2
        if self.dataset == 'cvact':
            s_root = self.root + '/satview_polish/'
        else:
            s_root = self.root + '/satellite/'

        s_path = self._get_pair_sample(s_root, _cls)
        s_img = self.loader(s_path)
        s_transform_ = transforms.Compose(self.s_transform)
        s_img = s_transform_(s_img)
        return img, s_img, target, weather

    def __len__(self):
        return len(self.imgs)
    
class MySatData_train(Data.Dataset):
    def __init__(self, root, transform=None, d_transform=None, loader=default_loader, view='/drone/',
                 style='normal',
                 dataset='university',
                 h=256, w=256, pad=10, erasing_p=0, color_jitter=True, DA=True):
        if dataset == 'cvact':
            sat_root = root + '/satview_polish/'
        else:
            sat_root = root + '/satellite/'
        classes, class_to_idx = find_classes(sat_root)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        imgs = make_pair_dataset(sat_root, class_to_idx, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        
        ###########################
        self.envir_list = [i for i in environments]
        self.style_list = self.envir_list + ['mixed']
        assert style in self.style_list, f"style must be one of {self.style_list}"
        # assert stage in ['train', 'test'], f"style must be one of {['train', 'test']}"
        self.style = style 
        ###################################

        transform_train_list = [
            # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
            transforms.Pad(pad, padding_mode='edge'),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        transform_satellite_list = [
            transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
            transforms.Pad(pad, padding_mode='edge'),
            transforms.RandomAffine(90),
            transforms.RandomCrop((h, w)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        transform_val_list = [
            transforms.Resize(size=(h, w), interpolation=3),  # Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        if color_jitter:
            transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                    hue=0)] + transform_train_list
            transform_satellite_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                        hue=0)] + transform_satellite_list

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform_satellite_list
        self.d_transform = transform_train_list
        self.loader = loader
        self.view = view

    def _get_pair_sample(self, root, _cls):
        img_path = []
        folder_root = root + str(_cls)
        assert os.path.isdir(folder_root), 'no pair drone image'
        for file_name in os.listdir(folder_root):
            img_path.append(folder_root + '/' + file_name)
        rand = np.random.permutation(len(img_path))
        tmp_index = rand[0]
        result_path = img_path[tmp_index]
        return result_path

    def __getitem__(self, index):
        """
        index (int): Index
    Returns:tuple: (image, target) where target is class_index of the target class.
        """
        # set weather 
        if self.style=='mixed':
            weather = np.random.choice(self.envir_list)
        else:
            weather = self.style

        transforms_d = self.d_transform
        if weather != 'normal':      
            #transform_d.insert(2, WeatherTransform(aug=weather))
            transforms_d = [None] + transforms_d
            transforms_d[0] = WeatherTransform(aug=weather)
            # print('APPENDED!')
        
        # load img 1
        transform_ = transforms.Compose(self.transform)
        path, _cls, target = self.imgs[index]
        img = self.loader(path)
        img = transform_(img)

        # load img 2
        d_root = self.root + self.view
        d_path = self._get_pair_sample(d_root, _cls)
        d_img = self.loader(d_path)
        d_transform_ = transforms.Compose(transforms_d)
        d_img = d_transform_(d_img)
        return img, d_img, target, weather

    def __len__(self):
        return len(self.imgs)
    
    

def init_dataset_train(data_dir, w=256, h=256, pad=10, batchsize=8, style='mixed', num_worker=4, num_worker_imgaug=0):

    data_transforms = {
        'train': transforms.Compose(
            [
                transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC), 
                transforms.Pad( pad, padding_mode='edge'),
                transforms.RandomCrop((h, w)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
        'val': transforms.Compose(
            [
                transforms.Resize(size=(h, w),interpolation=InterpolationMode.BICUBIC), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ),
    }   

    image_datasets = {
        'satellite': MySatData_train(data_dir, style=style, h=h, w=w, pad=pad),
        'street': datasets.ImageFolder(os.path.join(data_dir, 'street'), data_transforms['train']),
        'drone': MyDroneData_train(data_dir, style=style, h=h, w=w, pad=pad),
        'google': ImageFolder(os.path.join(data_dir, 'google'), data_transforms['train'])
    }

    dataloaders = {
        'satellite': torch.utils.data.DataLoader(image_datasets['satellite'], batch_size=batchsize, shuffle=True, num_workers=num_worker_imgaug, pin_memory=False),
        'street': torch.utils.data.DataLoader(image_datasets['street'], batch_size=batchsize, shuffle=True, num_workers=num_worker, pin_memory=False),
        'drone': torch.utils.data.DataLoader(image_datasets['drone'], batch_size=batchsize, shuffle=True, num_workers=num_worker_imgaug, pin_memory=False), # Must 0 here
        'google': torch.utils.data.DataLoader(image_datasets['google'], batch_size=batchsize, shuffle=True, num_workers=num_worker, pin_memory=False),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    return image_datasets, dataloaders, dataset_sizes

def init_dataset_test(name='University-Release', w=384, h=384, batchsize=128, style='mixed', num_worker=16):
    query_list = ['query_satellite', 'query_drone', 'query_street']
    gallery_list = ['gallery_satellite','gallery_drone', 'gallery_street']

    data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {x: MyDataset(os.path.join(os.getcwd(), name, 'test', x), style=style, stage='test',h=h, w=w)
                      if 'drone' in x 
                      else datasets.ImageFolder( os.path.join(name, 'test', x) ,data_transforms) 
                      for x in (query_list+gallery_list)}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize, shuffle=False, num_workers=0) 
                   if 'drone' in x 
                   else torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize, shuffle=False, num_workers=num_worker) 
                   for x in (query_list+gallery_list)}

    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    return image_datasets, dataloaders, dataset_sizes

environments2index = {env: idx for idx, env in enumerate(environments)}
index2environments = [i for i in environments]

def label2tensor(label, num_classes=10):
    idxs = [environments2index[i] for i in label]
    idxs = torch.tensor(idxs) 
    t = torch.nn.functional.one_hot(idxs, num_classes=num_classes).float()
    return t

def tensor2label(t):
    indices = torch.argmax(t, dim=1)
    return [index2environments[i] for i in indices]


    


