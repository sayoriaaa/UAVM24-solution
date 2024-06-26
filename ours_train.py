import os
import math
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.transforms import InterpolationMode
from torchvision import models

import numpy as np
import random

from utils.common import setup_seed 
# from utils.loader import environments, init_dataset_train
from torch import nn
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

from torch.nn import functional as F
from MLPN.loader import init_dataset_train
from utils.loader import  init_dataset_test, environments, tensor2label, label2tensor, init_dataset_test
from utils.metrics import metrics
from MLPN.model import CSWinTransv2_threeIn
from MLPN.utils import extract_feature, get_id, extract_feature, SAM, SupConLoss, one_LPN_output

from tqdm.contrib import tzip

import tarfile
import zipfile

from LPN.image_folder_ import CustomData160k_drone, CustomData160k_sat
from utils.competition import get_result_rank10, get_SatId_160k


class DomainClassifier(nn.Module):
    def __init__(self, domin_num=10) -> None:
        super(DomainClassifier, self).__init__()
        
        # adatped from modern backbone, change last fc to adopt domain
        self.net = models.resnet18(pretrained=True)
        ftr_num = self.net.fc.in_features
        self.net.fc = nn.Linear(ftr_num, domin_num)

    def forward(self, x):
        return self.net(x)
    

class WeatherFilter(nn.Module):
    def __init__(self, domin_num=10, h=256, w=256, channel=3) -> None:
        super(WeatherFilter, self).__init__()
        self.learnable_filters = nn.Parameter(torch.randn(domin_num, channel, h, w))

    def forward(self, weather_vec):
        weighted_filter = torch.einsum('bk,kchw->bcwh', weather_vec, self.learnable_filters)
        return weighted_filter
    
class FDM(nn.Module):
    def __init__(self, domin_num=10, h=256, w=256, channel=3) -> None:
        super(FDM, self).__init__()
        self.filter_invariant = WeatherFilter(domin_num=domin_num, h=h, w=w, channel=channel)
        self.filter_specific = WeatherFilter(domin_num=domin_num, h=h, w=w, channel=channel)

    def forward(self, img, vec):
        fft_img = torch.fft.fftn(img, dim=(-2,-1))
        amplitude = torch.abs(fft_img)
        phase = torch.angle(fft_img)

        fi = self.filter_invariant(vec)
        fs = self.filter_specific(vec)

        spec_invariant = fi * amplitude 
        spec_specific = fs * amplitude

        component_invariant = torch.real(torch.fft.ifftn(torch.polar(spec_invariant, phase), dim=(-2,-1)))
        component_specific = torch.real(torch.fft.ifftn(torch.polar(spec_specific, phase), dim=(-2,-1)))

        return component_invariant, component_specific

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin 

    def forward(self, out1, out2, label, block=4):
        # model out 2 tensor
        ff1 = out1[0]
        ff2 = out2[0]

        fnorm1 = torch.norm(ff1, p=2, dim=1, keepdim=True) * np.sqrt(block) 
        ff1 = ff1.div(fnorm1.expand_as(ff1)).view(ff1.size(0), -1)

        fnorm2 = torch.norm(ff2, p=2, dim=1, keepdim=True) * np.sqrt(block) 
        ff2 = ff2.div(fnorm2.expand_as(ff2)).view(ff2.size(0), -1)

        indices = torch.argmax(label, dim=1)
        batchsize = ff1.shape[0]
        loss = 0
        cnt = batchsize * batchsize - batchsize
        # ff1 is domain invariant feature
        # ff2 is domain specific feature


        for i in range(batchsize):
            for j in range(batchsize):
                if i==j:
                    continue 
                ed = nn.functional.pairwise_distance(ff1[i], ff2[j])
                if indices[i]==indices[j]: # positive sample
                    loss = loss + torch.pow(ed, 2)
                else: # negative sample
                    loss = loss + torch.pow(torch.clamp(self.margin-ed, min=0.0), 2)
        loss = loss / cnt 
        return loss 
    

class Ours:
    def __init__(self, 
                 use_wandb=True,
                 wandb_key = '16c9a3f92163ef4df08841029e02fded0cd0cfed'
                 ) -> None:
        self.seed = 2024
        self.use_wandb = use_wandb # use wandb to monitor training instead of CLI
        self.wandb_key = wandb_key
        # init
        setup_seed(self.seed)
        self.model_dir = os.path.join(os.getcwd(), 'model', 'Ours')
        os.environ['TORCH_HOME']='./' 
        
        # 
        # self.domain_classifer = DomainClassifier() 

    def train(self, 
              data_dir=None,
              style='mixed', 
              model_name1='FDM',
              model_name2='MLPN',
              num_epochs=210,
              lr = 0.005,
              batchsize = 8,
              block = 4,
              checkpoint_interval = 10,
              checkpoint_start = 40,
              droprate = 0.75,
              fix_img = True, # preprocess only once
              num_worker_imgaug = 16,
              margin = 1.0
              ):
        # use wandb to log
        if self.use_wandb:
            import wandb
            os.environ["WANDB_API_KEY"] = self.wandb_key
            wandb.init(project="ACMMMW24", name=model_name1)
        if data_dir==None:
            data_dir = os.path.join(os.getcwd(), 'University-Release', 'train')
        image_datasets, dataloaders, dataset_sizes = init_dataset_train(data_dir, batchsize=batchsize, style=style, num_worker_imgaug=num_worker_imgaug)

        # fix setting that
        # ====================
        # LPN: true
        # SAM: 1
        # balance: true 
        # infonce: 1
        # decouple: false
        # only_decouple: false 
        # moving_avg: 1.0
        # warm_epoch: 0 --> No Warmup
        # extra_Google: false
        # select_id: false
        # normal: false --> dataloader give couple

        model = CSWinTransv2_threeIn(701, droprate=droprate, decouple=False, infonce=1)
        model = model.cuda()
        model.train(True) 

        ignored_params = list()
        for i in range(block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            ignored_params += list(map(id, c.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optim_params = [{'params': base_params, 'lr': 0.1 * lr}]
        for i in range(block):
            cls_name = 'classifier' + str(i)
            c = getattr(model, cls_name)
            optim_params.append({'params': c.parameters(), 'lr': lr})

        infonce = SupConLoss(temperature=0.1)
        # SAM = 1
        base_optimizer = torch.optim.SGD
        optimizer_ft = SAM(optim_params, base_optimizer, lr=lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[120, 180, 210], gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        bestAcc, bestAp, bestEp = 0, 0, 0 

        #### OUR FDM Module
        fdm = FDM()
        fdm = fdm.cuda()
        fdm.train(True) 
        fdm_loss = ContrastiveLoss(margin=margin)


        if fix_img: # significant speed up, since imgaug is rather slow
            dataloader_drone = list()
            dataloader_sat = list()
            for data, data3 in tzip(dataloaders['drone'], dataloaders['satellite']):
                dataloader_drone.append(data)
                dataloader_sat.append(data3)
            print('done synthesis {} style image'.format(style))
        else:
            dataloader_drone = dataloaders['drone']
            dataloader_sat = dataloaders['satellite']


        for epoch in range(1, num_epochs+1):
            running_loss, running_corrects, running_corrects3 = 0.0, 0.0, 0.0
            ins_loss, dec_loss, on_loss, off_loss = 0.0, 0.0, 0.0, 0.0
            lossinfo1, lossinfo2 = 0.0, 0.0 
            lossfdm1, lossfdm2 = 0.0, 0.0
            optimizer = optimizer_ft
            # Iterate over data.
            for data, data3 in tzip(dataloader_sat, dataloader_drone):
                # get the inputs
                inputs, inputs_d, labels, weather = data
                inputs3, inputs3_s, labels3, weather3 = data3
                weather_vec = label2tensor(weather)
                weather_vec3 = label2tensor(weather3)
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < batchsize:  # skip the last batch
                    continue
                
                inputs = Variable(inputs.cuda().detach())
                inputs_d = Variable(inputs_d.cuda().detach())
                inputs3 = Variable(inputs3.cuda().detach())
                inputs3_s = Variable(inputs3_s.cuda().detach())
                labels = Variable(labels.cuda().detach())
                labels3 = Variable(labels3.cuda().detach())
                weather_vec = Variable(weather_vec.cuda().detach())
                weather_vec3 = Variable(weather_vec3.cuda().detach())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward FDM
                inputs_d_ci, inputs_d_cs = fdm(inputs_d, weather_vec)
                inputs3_ci, inputs3_cs = fdm(inputs3, weather_vec3)

                # forward (invariant part and satellite img)
                outs_c, outs_info = model(inputs) # decouple: false
                outs_d_c, outs_d_info = model(inputs_d_ci) # balance: true and infonce: 1 

                outd_c, outd_info = model(inputs3_ci)
                outs3_s_c, outs3_s_info = model(inputs3_s)

                # calculate loss 
                preds, loss = one_LPN_output(outs_c, labels, criterion, block)
                _, loss_d = one_LPN_output(outs_d_c, labels, criterion, block)
                loss = loss + loss_d
                preds3, loss3 = one_LPN_output(outd_c, labels3, criterion, block)
                _, loss3_s = one_LPN_output(outs3_s_c, labels3, criterion, block)
                loss3 = loss3 + loss3_s
                loss = (loss + loss3) / 2

                sate = F.normalize(outs_info, dim=1)
                drone = F.normalize(outd_info, dim=1)
                sate_ = F.normalize(outs_d_info, dim=1)
                drone_ = F.normalize(outs3_s_info, dim=1)
                features1 = torch.cat([sate.unsqueeze(1), sate_.unsqueeze(1)], dim=1)
                features2 = torch.cat([drone.unsqueeze(1), drone_.unsqueeze(1)], dim=1)
                loss_info = infonce(features1, labels)
                loss = loss + loss_info
                loss_info1 = infonce(features2, labels3)
                loss = loss + loss_info1

                # fdm (specific part, for FDM)
                outs_d_c_s, _ = model(inputs_d_cs)
                outd_c_s, _ = model(inputs3_cs)

                fdm_inputs_loss = fdm_loss(outs_d_c, outs_d_c_s, weather_vec)
                fdm_inputs3_loss = fdm_loss(outd_c, outd_c_s, weather_vec3)
                loss = loss + fdm_inputs_loss + fdm_inputs3_loss

                # backward
                loss.backward()
                optimizer.first_step(zero_grad=True)

                # again
                # forward FDM
                inputs_d_ci, inputs_d_cs = fdm(inputs_d, weather_vec)
                inputs3_ci, inputs3_cs = fdm(inputs3, weather_vec3)

                # forward (invariant part and satellite img)
                outs_c, outs_info = model(inputs) # decouple: false
                outs_d_c, outs_d_info = model(inputs_d_ci) # balance: true and infonce: 1 

                outd_c, outd_info = model(inputs3_ci)
                outs3_s_c, outs3_s_info = model(inputs3_s)

                # calculate loss 
                preds, loss = one_LPN_output(outs_c, labels, criterion, block)
                _, loss_d = one_LPN_output(outs_d_c, labels, criterion, block)
                loss = loss + loss_d
                preds3, loss3 = one_LPN_output(outd_c, labels3, criterion, block)
                _, loss3_s = one_LPN_output(outs3_s_c, labels3, criterion, block)
                loss3 = loss3 + loss3_s
                loss = (loss + loss3) / 2

                sate = F.normalize(outs_info, dim=1)
                drone = F.normalize(outd_info, dim=1)
                sate_ = F.normalize(outs_d_info, dim=1)
                drone_ = F.normalize(outs3_s_info, dim=1)
                features1 = torch.cat([sate.unsqueeze(1), sate_.unsqueeze(1)], dim=1)
                features2 = torch.cat([drone.unsqueeze(1), drone_.unsqueeze(1)], dim=1)
                loss_info = infonce(features1, labels)
                loss = loss + loss_info
                loss_info1 = infonce(features2, labels3)
                loss = loss + loss_info1

                # fdm (specific part, for FDM)
                outs_d_c_s, _ = model(inputs_d_cs)
                outd_c_s, _ = model(inputs3_cs)

                fdm_inputs_loss = fdm_loss(outs_d_c, outs_d_c_s, weather_vec)
                fdm_inputs3_loss = fdm_loss(outd_c, outd_c_s, weather_vec3)
                loss = loss + fdm_inputs_loss + fdm_inputs3_loss

                loss.backward()
                optimizer.second_step(zero_grad=True)

                # statistics
                running_loss += loss.item() * now_batch_size 
                lossinfo1 += loss_info.item() * now_batch_size
                lossfdm1 += fdm_inputs_loss.item() * now_batch_size 
                lossfdm2 += fdm_inputs3_loss.item() * now_batch_size 
                running_corrects += float(torch.sum(preds == labels.data))
                running_corrects3 += float(torch.sum(preds3 == labels3.data))

            epoch_loss = running_loss / dataset_sizes['satellite']
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc3 = running_corrects3 / dataset_sizes['satellite']

            epoch_loss_info1 = lossinfo1 / dataset_sizes['satellite']
            epoch_loss_fdm1 = lossfdm1 / dataset_sizes['satellite']
            epoch_loss_fdm2 = lossfdm2 / dataset_sizes['satellite']

            print('{} Loss: {:.4f} Satellite_Acc: {:.4f} Drone_Acc: {:.4f} infoloss1: {:.4f} fdm1loss: {:.4f} fdm2loss: {:.4f} '.format(
                epoch, epoch_loss, epoch_acc,
                epoch_acc3, epoch_loss_info1, epoch_loss_fdm1, epoch_loss_fdm2))
            
            if self.use_wandb:
                wandb.log({'Loss': epoch_loss, 
                            'Satellite_Acc': epoch_acc,
                            'Drone_Acc': epoch_acc3,
                            'infoloss1': epoch_loss_info1,
                            'fdm1loss': epoch_loss_fdm1,
                            'fdm2loss': epoch_loss_fdm2
                            })
            
            exp_lr_scheduler.step()

            if epoch % checkpoint_interval == 0 and epoch > checkpoint_start:
                
                save_filename = '{}_{:03d}.pth'.format(model_name1, epoch)
                save_path = os.path.join(os.path.join(self.model_dir, save_filename))
                torch.save(fdm.cpu().state_dict(), save_path)
                fdm.cuda()

                save_filename = '{}_{:03d}.pth'.format(model_name2, epoch)
                save_path = os.path.join(os.path.join(self.model_dir, save_filename))
                torch.save(model.cpu().state_dict(), save_path)
                model.cuda() # essential!  

                


if __name__ == '__main__':
    m = Ours()
    # m.train_multi_MLPNs(data_dir='/dataset/University-Release/train')
    #style='normal'
    #m.train(data_dir='/dataset/University-Release/train', style=style, model_name=style, num_epochs=20, checkpoint_interval=5, num_worker_imgaug=32, fix_img=True)
    #style='rain'
    #m.train(data_dir='/dataset/University-Release/train', style=style, model_name=style, num_epochs=20, checkpoint_interval=5, num_worker_imgaug=32, fix_img=True)
    style='mixed'
    m.train(data_dir='/dataset/University-Release/train')