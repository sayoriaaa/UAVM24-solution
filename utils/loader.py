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
    

class MyDataset(datasets.ImageFolder):
    # enable various weather enhancement
    # use for drone-view only
    def __init__(self, root, transform = None, target_transform = None, style='normal', stage='train', h=384, w=384, pad=10):
        super().__init__(root, transform = transform, target_transform = target_transform)
        self.envir_list = [i for i in environments]
        self.style_list = self.envir_list + ['mixed']
        assert style in self.style_list, f"style must be one of {self.style_list}"
        assert stage in ['train', 'test'], f"style must be one of {['train', 'test']}"
        self.style = style 
        self.stage = stage
        # transform setting
        self.h = h 
        self.w = w 
        self.pad = pad 

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # enhance image according to self.style 
        h = self.h 
        w = self.w 
        pad = self.pad 

        if self.style=='mixed':
            weather = np.random.choice(self.envir_list)
        else:
            weather = self.style


        if self.stage=='train':
            t = transforms.Compose(
                [
                    transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC), 
                    WeatherTransform(aug=weather),
                    transforms.Pad( pad, padding_mode='edge'),
                    transforms.RandomCrop((h, w)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        else:
            t = transforms.Compose(
                [
                    transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
                    WeatherTransform(aug=weather),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        
        img = t(img)
        return img, target, weather    

    def __len__(self):
        return len(self.imgs)
    

def init_dataset_train(name='University-Release', w=384, h=384, pad=10, batchsize=8, style='mixed', num_worker=4):

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
        'satellite': transforms.Compose(
            [
                transforms.Resize((h, w), interpolation=InterpolationMode.BICUBIC),
                transforms.Pad( pad, padding_mode='edge'),
                transforms.RandomAffine(90),
                transforms.RandomCrop((h, w)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        ) 
    }   

    image_datasets = {
        'satellite': datasets.ImageFolder(os.path.join(os.getcwd(), name, 'train', 'satellite'), data_transforms['satellite']),
        'street': datasets.ImageFolder(os.path.join(os.getcwd(), name, 'train', 'street'), data_transforms['train']),
        'drone': MyDataset(os.path.join(os.getcwd(), name, 'train', 'drone'), style=style, stage='train',h=h, w=w, pad=pad),
        'google': ImageFolder(os.path.join(os.getcwd(), name, 'train', 'google'), data_transforms['train'])
    }

    dataloaders = {
        'satellite': torch.utils.data.DataLoader(image_datasets['satellite'], batch_size=batchsize, shuffle=True, num_workers=num_worker, pin_memory=False),
        'street': torch.utils.data.DataLoader(image_datasets['street'], batch_size=batchsize, shuffle=True, num_workers=num_worker, pin_memory=False),
        'drone': torch.utils.data.DataLoader(image_datasets['drone'], batch_size=batchsize, shuffle=True, num_workers=0, pin_memory=False), # Must 0 here
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
    


