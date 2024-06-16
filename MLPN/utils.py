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

def extract_feature(model, dataloaders, view = 'satellite', testing=False, block=4, LPN=True, decouple=True, infonce=1, ms=[1]):
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
                """
                if view == 'satellite':
                    outputs, _, _ = model(input_img, None, None)
                elif view == 'street':
                    _, outputs, _ = model(None, input_img, None)
                elif view == 'drone':
                    _, _, outputs = model(None, None, input_img)
                """
                if view == 'street':
                    _, outputs, _ = model(None, input_img, None)
                else:
                    outputs = model(input_img)
                ff += outputs[0]

        if LPN:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(block) 
            ff = ff.div(fnorm.expand_as(ff)).view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features
            



# work channel loss
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def decouple_loss(y1, y2, scale_loss, lambd):
    batch_size = y1.size(0)
    c = y1.T @ y2
    c.div_(batch_size)
    on_diag = torch.diagonal(c)
    p_on = (1 - on_diag) / 2
    on_diag = torch.pow(p_on, opt.e1) * torch.pow(torch.diagonal(c).add_(-1), 2)
    on_diag = on_diag.sum().mul(scale_loss)

    off_diag = off_diagonal(c)
    p_off = torch.abs(off_diag)
    off_diag = torch.pow(p_off, opt.e2) * torch.pow(off_diagonal(c), 2)
    off_diag = off_diag.sum().mul(scale_loss)
    loss = on_diag + off_diag * lambd
    return loss, on_diag, off_diag * lambd


def one_LPN_output(outputs, labels, criterion, block):
    # part = {}
    sm = nn.Softmax(dim=1)
    num_part = block
    score = 0
    loss = 0
    for i in range(num_part):
        part = outputs[i]
        score += sm(part)
        loss += criterion(part, labels)

    _, preds = torch.max(score.data, 1)

    return preds, loss


def one_info_output(Soutputs, Doutputs, labels, criterion, block):
    num_part = block
    for i in range(num_part):
        Dpart = Soutputs[i]  # 2,701
        Spart = Doutputs[i]  # 2,701
        s_norm = F.normalize(Dpart, dim=1)
        d_norm = F.normalize(Spart, dim=1)
        features = torch.cat([s_norm, d_norm], dim=1)  # 2,701