import torch 
import os 
import torch.nn as nn
import math
import numpy as np

from tqdm import tqdm
from tqdm.contrib import tzip
from torch.utils.data import Subset, DataLoader
from torch.autograd import Variable

from utils.loader import label2tensor

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

def extract_feature(DC_model, FFM_model, MLPN_model, dataloaders, view = 'satellite', testing=True, block=4, decouple=True, infonce=1, ms=[1]):
    features = torch.FloatTensor()
    for data in tqdm(dataloaders, desc='Extract {} feature'.format(view)):
        if view == 'drone':
            if testing:
                img, label = data 
                img_ = img.cuda()
                logits = DC_model(img_)
                weather = torch.softmax(logits, dim=1)
            else:
                img, label, weather = data 
                weather = label2tensor(weather)
        else:
            img, label = data 
        n, c, h, w = img.size()
        
        ff = torch.FloatTensor(n,512,block).zero_().cuda()

        for i in [img, fliplr(img)]:
            for scale in ms:
                input_img = scale_img(i, scale).cuda()# .requires_grad_()
                if view == 'street':
                    _, outputs, _ = MLPN_model(None, input_img, None)
                elif view == 'drone':
                    input_img_i = FFM_model(input_img, weather)
                    outputs = MLPN_model(input_img_i)
                else:
                    outputs = MLPN_model(input_img)
                ff += outputs[0]

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(block) 
        ff = ff.div(fnorm.expand_as(ff)).view(ff.size(0), -1)

        features = torch.cat((features,ff.data.cpu()), 0)
    return features
            



# work channel loss
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def decouple_loss(y1, y2, scale_loss, lambd, e1 = 1, e2 = 1):
    batch_size = y1.size(0)
    c = y1.T @ y2
    c.div_(batch_size)
    on_diag = torch.diagonal(c)
    p_on = (1 - on_diag) / 2
    on_diag = torch.pow(p_on, e1) * torch.pow(torch.diagonal(c).add_(-1), 2)
    on_diag = on_diag.sum().mul(scale_loss)

    off_diag = off_diagonal(c)
    p_off = torch.abs(off_diag)
    off_diag = torch.pow(p_off, e2) * torch.pow(off_diagonal(c), 2)
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

def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    toogle_grad(model_src, True)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # print(self.state[p]["old_p"],"1233333333333")
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels_column = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            #mask = torch.eq(labels, labels.T).float().to(device)
            mask = (labels_column == labels).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # print(contrast_count,"contrast_count")
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(contrast_feature.shape,"contrast_feature")

        # print(f"contrast_feature {contrast_feature.size()}")
        # print(f"mask {mask.size()}")
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print(f"mask repeat {mask.size()}")
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # print(f"logits {logits.size()}")
        # assert(0)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss