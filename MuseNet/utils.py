# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch 
import os 
import torch.nn as nn
import math
import numpy as np

from tqdm import tqdm
from tqdm.contrib import tzip
from torch.utils.data import Subset, DataLoader
from torch.autograd import Variable


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
    return labels

def scale_img(img, scale_factor):
    if scale_factor==1:
        return img 
    scale_factor = math.sqrt(scale_factor)
    return nn.functional.interpolate(img, scale_factor=scale_factor, mode='bilinear', align_corners=False)

def extract_feature(model, dataloaders, view = 'satellite', testing=False, block=4, LPN=True, ms=[1]):
    features = torch.FloatTensor()
    for data in tqdm(dataloaders, desc='Extract {} feature'.format(view)):
        if view == 'drone' and not testing:
            img, label, weather = data 
        else:
            img, label = data 
        n, c, h, w = img.size()

        if LPN:
            ff = torch.FloatTensor(n,512,block).zero_().cuda()
        else:
            ff = torch.FloatTensor(n,512).zero_().cuda()

        for i in [img, fliplr(img)]:
            for scale in ms:
                input_img = scale_img(i, scale).cuda()# .requires_grad_()
                if view == 'satellite':
                    outputs, _, _ = model(input_img, None, None)
                elif view == 'street':
                    _, outputs, _ = model(None, input_img, None)
                elif view == 'drone':
                    _, _, outputs = model(None, None, input_img)
                ff += outputs

        if LPN:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(block) 
            ff = ff.div(fnorm.expand_as(ff)).view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features


