import torch.utils.data as Data
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from os.path import join as ospj
from PIL import Image
import os
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


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
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

#----------------------------------------------------------------------------#
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
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, rotate = 0, pad = 0):
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.webp']
        imgs = make_dataset_160k_sat(root, IMG_EXTENSIONS)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

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
            img = transforms.functional.rotate(img,self.rotate)
            if self.pad > 0:
                img = transforms.functional.resize(img,(256,256),interpolation=3)
                img = transforms.functional.pad(img,(self.pad,0,0,0))
                img = transforms.functional.five_crop(img,(256,256))[0]
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
    def __init__(self, root, transform = None, target_transform = None, loader = default_loader, query_name = None, rotate = 0, pad = 0):
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.webp']
        imgs = make_dataset_160k_drone(root, IMG_EXTENSIONS, query_name)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

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
            img = transforms.functional.rotate(img,self.rotate)
            if self.pad > 0:
                img = transforms.functional.resize(img,(256,256),interpolation=3)
                img = transforms.functional.pad(img,(self.pad,0,0,0))
                img = transforms.functional.five_crop(img,(256,256))[0]
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