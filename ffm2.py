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
from FFM.model import CSWinTransv2_threeIn
from FFM.utils import extract_feature, get_id, extract_feature, SAM, SupConLoss, one_LPN_output

from tqdm.contrib import tzip

import tarfile
import zipfile

from LPN.image_folder_ import CustomData160k_drone, CustomData160k_sat
from utils.competition import get_result_rank10, get_SatId_160k

    

class FFilter(nn.Module):
    def __init__(self, domin_num=10, h=256, w=256, channel=3) -> None:
        super(FFilter, self).__init__()
        self.learnable_filter = nn.Parameter(torch.ones(channel, h, w))

    def forward(self, x):
        return self.learnable_filter * x
    
class FFM(nn.Module):
    def __init__(self, domin_num=10, h=256, w=256, channel=3) -> None:
        super(FFM, self).__init__()
        self.filter_invariant = FFilter(domin_num=domin_num, h=h, w=w, channel=channel)

    def forward(self, img):
        fft_img = torch.fft.fftn(img, dim=(-2,-1))
        amplitude = torch.abs(fft_img)
        phase = torch.angle(fft_img)

        spec_invariant = self.filter_invariant(amplitude)
        component_invariant = torch.real(torch.fft.ifftn(torch.polar(spec_invariant, phase), dim=(-2,-1)))
        return component_invariant
    
    
def get_fixed_dataloader(dataloaders):
    dataloader_drone = list()
    dataloader_sat = list()
    for data, data3 in tzip(dataloaders['drone'], dataloaders['satellite']):
        dataloader_drone.append(data)
        dataloader_sat.append(data3)
    print('done synthesis {} style image'.format(style))
    return dataloader_drone, dataloader_sat
    

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
        self.model_dir = os.path.join(os.getcwd(), 'model', 'FFMv2')
        os.environ['TORCH_HOME']='./' 

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)    
        # 
        # self.domain_classifer = DomainClassifier() 

    def train(self, 
              data_dir=None,
              style='mixed', 
              model_name1='FFMv2',
              model_name2='MLPN',
              num_epochs=210,
              lr = 0.005,
              ffm_lr = 0.01,
              batchsize = 8,
              block = 4,
              checkpoint_interval = 10,
              checkpoint_start = 40,
              droprate = 0.75,
              update_aug_img = [40, 80, 120, 160, 180, 200], 
              num_worker_imgaug = 16,
              opt_iter_epoch = 32
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

        #### MLPN 
        model = CSWinTransv2_threeIn(701, droprate=droprate, decouple=False, infonce=1)
        model = model.cuda()

        #### OUR FFM Module
        ffm_sat = FFM()
        ffm_sat = ffm_sat.cuda()
        ffm_drone = FFM()
        ffm_drone = ffm_drone.cuda()

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
        
        optim_params.append({'params': ffm_drone.parameters(), 'lr': ffm_lr})
        optim_params.append({'params': ffm_sat.parameters(), 'lr': ffm_lr})

        infonce = SupConLoss(temperature=0.1)
        # SAM = 1
        base_optimizer = torch.optim.SGD
        optimizer_ft = SAM(optim_params, base_optimizer, lr=lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[120, 180, 210], gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        bestAcc, bestAp, bestEp = 0, 0, 0 

        dataloader_drone, dataloader_sat = get_fixed_dataloader(dataloaders)
        optimizer = optimizer_ft

        for epoch in range(1, num_epochs+1):
            running_loss, running_corrects, running_corrects3 = 0.0, 0.0, 0.0
            ins_loss, dec_loss, on_loss, off_loss = 0.0, 0.0, 0.0, 0.0
            lossinfo1, lossinfo2 = 0.0, 0.0 
                    
            if epoch in update_aug_img:
                dataloader_drone, dataloader_sat = get_fixed_dataloader(dataloaders)

            

            # Iterate over data.
            for data, data3 in tzip(dataloader_sat, dataloader_drone):
                # get the inputs
                inputs, inputs_d, labels, weather = data
                inputs3, inputs3_s, labels3, weather3 = data3

                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < batchsize:  # skip the last batch
                    continue
                
                inputs = Variable(inputs.cuda().detach())
                inputs_d = Variable(inputs_d.cuda().detach())
                inputs3 = Variable(inputs3.cuda().detach())
                inputs3_s = Variable(inputs3_s.cuda().detach())
                labels = Variable(labels.cuda().detach())
                labels3 = Variable(labels3.cuda().detach())


                # zero the parameter gradients
                optimizer.zero_grad()
                inputs = ffm_sat(inputs)
                inputs3_s = ffm_sat(inputs3_s)
                inputs_d = ffm_drone(inputs_d)
                inputs3 = ffm_drone(inputs3)

                # forward (invariant part and satellite img)
                outs_c, outs_info = model(inputs) # decouple: false
                outs_d_c, outs_d_info = model(inputs_d) # balance: true and infonce: 1 

                outd_c, outd_info = model(inputs3)
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

                # backward
                loss.backward(retain_graph=True)
                optimizer.first_step(zero_grad=True)

                # again
                # forward ffm
                inputs = ffm_sat(inputs)
                inputs3_s = ffm_sat(inputs3_s)
                inputs_d = ffm_drone(inputs_d)
                inputs3 = ffm_drone(inputs3)

                # forward (invariant part and satellite img)
                outs_c, outs_info = model(inputs) # decouple: false
                outs_d_c, outs_d_info = model(inputs_d) # balance: true and infonce: 1 

                outd_c, outd_info = model(inputs3)
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

                # backward
                loss.backward()
                optimizer.second_step(zero_grad=True)

                # statistics
                running_loss += loss.item() * now_batch_size 
                lossinfo1 += loss_info.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
                running_corrects3 += float(torch.sum(preds3 == labels3.data))


            epoch_loss = running_loss / dataset_sizes['satellite']
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc3 = running_corrects3 / dataset_sizes['satellite']

            epoch_loss_info1 = lossinfo1 / dataset_sizes['satellite']

            print('{} Loss: {:.4f} Satellite_Acc: {:.4f} Drone_Acc: {:.4f} infoloss1: {:.4f} '.format(
                epoch, epoch_loss, epoch_acc,
                epoch_acc3, epoch_loss_info1))
            
            if self.use_wandb:
                wandb.log({'Loss': epoch_loss, 
                            'Satellite_Acc': epoch_acc,
                            'Drone_Acc': epoch_acc3,
                            'infoloss1': epoch_loss_info1
                            })
            
            exp_lr_scheduler.step()

            if epoch % checkpoint_interval == 0 and epoch > checkpoint_start:
                
                save_filename = '{}_drone_{:03d}.pth'.format(model_name1, epoch)
                save_path = os.path.join(os.path.join(self.model_dir, save_filename))
                torch.save(ffm_drone.cpu().state_dict(), save_path)
                ffm_drone.cuda()

                save_filename = '{}_sat_{:03d}.pth'.format(model_name1, epoch)
                save_path = os.path.join(os.path.join(self.model_dir, save_filename))
                torch.save(ffm_sat.cpu().state_dict(), save_path)
                ffm_sat.cuda()

                save_filename = '{}_{:03d}.pth'.format(model_name2, epoch)
                save_path = os.path.join(os.path.join(self.model_dir, save_filename))
                torch.save(model.cpu().state_dict(), save_path)
                model.cuda() # essential!  

    def get_competition_submit(self, data160k_dir='D://dataset/university-160k-wx', save_file = 'answer.txt', pth=None, multiple_scale=[1], batchsize=128, block=4, MLPN_file='MLPN_200.pth', DC_file='DomainClassifier_060.pth', FFM_file='FFM_200.pth'):
        # a part of test set of UAVM'24 competition is provided as tar file
        if not os.path.exists(os.path.join(data160k_dir, 'gallery_satellite_160k')):
            tar_file = os.path.join(data160k_dir, 'gallery_satellite_160k.tar.gz')
            if os.path.isfile(tar_file):
                print('Found dataset tar file. Extracting...')
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.extractall(path=data160k_dir)
                print('Extract done')
        else:
            print('Found dataset')

        query_name = os.path.join(data160k_dir, 'query_drone_name.txt')
        if os.path.isfile(save_file):
            os.remove(save_file) 
            os.remove('answer.zip')
        results_rank10 = [] 

        data_transforms = transforms.Compose([
            transforms.Resize((self.h, self.w), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_datasets = {}
        image_datasets['gallery_satellite_160k'] = CustomData160k_sat(os.path.join(data160k_dir, 'gallery_satellite_160k'), data_transforms)
        image_datasets['query_drone_160k'] = CustomData160k_drone( os.path.join(data160k_dir, 'query_drone160k_wx') ,data_transforms, query_name = query_name)

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                        shuffle=False, num_workers=16) for x in
                        ['gallery_satellite_160k','query_drone_160k']}
        
        gallery_path = image_datasets['gallery_satellite_160k'].imgs
        gallery_label, gallery_path  = get_SatId_160k(gallery_path)
        
        # load model
        ## load MLPN model
        print("load MLPN model: {}".format(MLPN_file))
        MLPN_file = os.path.join(self.model_dir, MLPN_file)
        MLPN_model = CSWinTransv2_threeIn(701, droprate=0.75, decouple=False, infonce=1)
        MLPN_model.load_state_dict(torch.load(MLPN_file)) 
        # LPN: true
        for i in range(block):
            cls_name = 'classifier'+str(i)
            c = getattr(MLPN_model, cls_name)
            c.classifier = nn.Sequential()
        MLPN_model = MLPN_model.cuda()
        MLPN_model.train(False)
        ## load Domain Classifier model
        DC_file = os.path.join(self.model_dir, DC_file)
        domain_classifier = DomainClassifier()
        domain_classifier.load_state_dict(torch.load(DC_file)) 
        domain_classifier = domain_classifier.cuda()
        domain_classifier.train(False)
        ## load FFM model
        FFM_file = os.path.join(self.model_dir, FFM_file)
        ffm = FFM()
        ffm.load_state_dict(torch.load(FFM_file)) 
        ffm = ffm.cuda()
        ffm.train(False)

        # Extract features
        with torch.no_grad():
            query_feature = extract_feature(domain_classifier,ffm, MLPN_model, dataloaders['query_drone_160k'], view='drone', ms=multiple_scale, testing=True)
            gallery_feature = extract_feature(domain_classifier,ffm, MLPN_model, dataloaders['gallery_satellite_160k'], view='satellite', ms=multiple_scale)
        
        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

        gallery_label = np.array(gallery_label)
        for i in tqdm(range(len(query_feature)), desc='Evaluate Rank10 results'):
            result_rank10 = get_result_rank10(query_feature[i], gallery_feature, gallery_label)
            results_rank10.append(result_rank10)
            
        results_rank10 = np.row_stack(results_rank10)
        with open(save_file, 'w') as f:
            for row in results_rank10:
                f.write('\t'.join(map(str, row)) + '\n')

        # zip
        zip_name = os.path.join(os.getcwd(), 'answer.zip')
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(save_file, save_file)            


if __name__ == '__main__':
    m = Ours()
    # m.train_multi_MLPNs(data_dir='/dataset/University-Release/train')
    #style='normal'
    #m.train(data_dir='/dataset/University-Release/train', style=style, model_name=style, num_epochs=20, checkpoint_interval=5, num_worker_imgaug=32, fix_img=True)
    #style='rain'
    #m.train(data_dir='/dataset/University-Release/train', style=style, model_name=style, num_epochs=20, checkpoint_interval=5, num_worker_imgaug=32, fix_img=True)
    style='mixed'
    m.train(data_dir='/dataset/University-Release/train', update_aug_img=list(range(10,200)), checkpoint_interval=10, checkpoint_start=0)