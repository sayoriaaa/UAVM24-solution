import os
import torch
import torch.nn as nn 
from torch.nn import functional as F
from utils.common import setup_seed 
from MLPN.loader import init_dataset_train
from utils.loader import  init_dataset_test, environments
from utils.metrics import metrics
from MLPN.model import CSWinTransv2_threeIn
from MLPN.utils import extract_feature, get_id, extract_feature, SAM, SupConLoss, one_LPN_output
from torch.autograd import Variable

from torch.optim import lr_scheduler
from tqdm.contrib import tzip

from utils.competition import get_result_rank10, get_SatId_160k
from LPN.image_folder_ import CustomData160k_drone, CustomData160k_sat
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import numpy as np
from tqdm import tqdm
import tarfile
import zipfile



def get_fixed_dataloader(dataloaders, style='mixed'):
    dataloader_drone = list()
    dataloader_sat = list()
    for data, data3 in tzip(dataloaders['drone'], dataloaders['satellite']):
        dataloader_drone.append(data)
        dataloader_sat.append(data3)
    print('done synthesis {} style image'.format(style))
    return dataloader_drone, dataloader_sat

class MLPN_:
    def __init__(self, 
                 use_wandb=True,
                 wandb_key = '16c9a3f92163ef4df08841029e02fded0cd0cfed') -> None:
        # default parameters
        self.seed = 2024
        self.use_wandb = use_wandb # use wandb to monitor training instead of CLI
        self.wandb_key = wandb_key

        self.pytorch_version =  torch.__version__      
        self.w = 256
        self.h = 256

        # init
        setup_seed(self.seed)
        self.model_dir = os.path.join(os.getcwd(), 'model', 'MLPN')
        os.environ['TORCH_HOME']='./'  

        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)    

    def train(self, 
              data_dir=None,
              style='mixed', 
              model_name='net',
              num_epochs=210,
              lr = 0.005,
              batchsize = 8,
              block = 4,
              checkpoint_interval = 10,
              checkpoint_start = 0,
              droprate = 0.75,
              update_aug_img = [40, 80, 120, 160, 180, 200],
              num_worker_imgaug = 16,
              ):
        if self.use_wandb:
            import wandb
            os.environ["WANDB_API_KEY"] = self.wandb_key
            wandb.init(project="ACMMMW24", name=model_name)

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

        dataloader_drone, dataloader_sat = get_fixed_dataloader(dataloaders)

        for epoch in range(1, num_epochs+1):
            if epoch in update_aug_img:
                dataloader_drone, dataloader_sat = get_fixed_dataloader(dataloaders)
            running_loss, running_corrects, running_corrects3 = 0.0, 0.0, 0.0
            ins_loss, dec_loss, on_loss, off_loss = 0.0, 0.0, 0.0, 0.0
            lossinfo1, lossinfo2 = 0.0, 0.0 
            optimizer = optimizer_ft
            # Iterate over data.
            for data, data3 in tzip(dataloader_sat, dataloader_drone):
                # get the inputs
                inputs, inputs_d, labels, _ = data
                inputs3, inputs3_s, labels3, _ = data3
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

                # forward
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
                optimizer.first_step(zero_grad=True)

                # again
                outd_c, outd_info = model(inputs3)
                outs_c, outs_info = model(inputs)
                outs_d_c, outs_d_info = model(inputs_d)
                outs3_s_c, outs3_s_info = model(inputs3_s)

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

            if self.use_wandb:
                wandb.log({'Loss': epoch_loss, 
                            'Satellite_Acc': epoch_acc,
                            'Drone_Acc': epoch_acc3,
                            'infoloss1': epoch_loss_info1
                            })

            print('{} Loss: {:.4f} Satellite_Acc: {:.4f} Drone_Acc: {:.4f} infoloss1: {:.4f} infoloss2: {:.4f}'.format(
                epoch, epoch_loss, epoch_acc,
                epoch_acc3, epoch_loss_info1, 0.00))
            
            exp_lr_scheduler.step()

            if epoch % checkpoint_interval == 0 and epoch > checkpoint_start:     
                save_filename = '{}_{:03d}.pth'.format(model_name, epoch)
                save_path = os.path.join(os.path.join(self.model_dir, save_filename))
                torch.save(model.cpu().state_dict(), save_path)
                model.cuda() # essential!   
        



    def train_multi_MLPNs(self, data_dir=None):
        # For competition
        self.model_dir = os.path.join(os.getcwd(), 'model', 'MLPNs')
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        for style in environments:
            if style=='normal':
                continue
            print(style)
            self.train(data_dir=data_dir, style=style, model_name=style, num_epochs=120, checkpoint_interval=20, checkpoint_start=100, num_worker_imgaug=32, fix_img=True)

        



    def test(self, pth=None, query='drone', gallery='satellite', multiple_scale=[1], batchsize=128, style='mixed'):
        # load data
        image_datasets, dataloaders, dataset_sizes = init_dataset_test(batchsize=batchsize, style=style, w=self.w, h=self.h)
        # init label
        gallery_name = 'gallery_' + gallery
        query_name = 'query_' + query 
        gallery_label = get_id(image_datasets[gallery_name].imgs)
        query_label = get_id(image_datasets[query_name].imgs)
        # print(dataset_sizes[gallery_name])
        # load model
        model_file = os.listdir(self.model_dir)[-1] if pth==None else pth + '.pth'
        print("load model: {}".format(model_file))
        model = CSWinTransv2_threeIn(701, droprate=0.75, decouple=False, infonce=1)
        

        try:
            model.load_state_dict(torch.load(os.path.join(self.model_dir, model_file))) 
        except:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(self.model_dir, model_file))) 
            model = model.module

        if self.LPN:
            for i in range(self.block):
                cls_name = 'classifier'+str(i)
                c = getattr(model, cls_name)
                c.classifier = nn.Sequential()
        else:
            model.model.classifier.classifier = nn.Sequential()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = model.to(device)
        # model.model = model.model.to(device)
        model = model.eval()

        # extract features
        with torch.no_grad():
            query_feature = extract_feature(model,dataloaders[query_name], view=query, ms=multiple_scale)
            gallery_feature = extract_feature(model,dataloaders[gallery_name], view=gallery, ms=multiple_scale)

        # calculate 
        m = metrics(query_feature, query_label, gallery_feature, gallery_label)
        print("Recall@1: {:.2f}".format(m[0]))
        print("Recall@5: {:.2f}".format(m[1]))
        print("Recall@10: {:.2f}".format(m[2]))
        print("Recall@top1: {:.2f}".format(m[3]))
        print("Recall@AP: {:.2f}".format(m[4]))
        # return m

    def load_model(self, pth=None, block=4):
        model_file = os.listdir(self.model_dir)[-1] if pth==None else pth + '.pth'
        print("load model: {}".format(model_file))
        model = CSWinTransv2_threeIn(701, droprate=0.75, decouple=False, infonce=1)

        try:
            model.load_state_dict(torch.load(os.path.join(self.model_dir, model_file))) 
        except:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(os.path.join(self.model_dir, model_file))) 
            model = model.module
        # LPN: true
        for i in range(block):
            cls_name = 'classifier'+str(i)
            c = getattr(model, cls_name)
            c.classifier = nn.Sequential()

        model = model.cuda()
        model.train(False)
        return model 
    
    def get_competition_submit(self, data160k_dir='D://dataset/university-160k-wx', save_file = 'answer.txt', pth=None, multiple_scale=[1], batchsize=128):
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
        model = self.load_model(pth=pth)

        # Extract features
        with torch.no_grad():
            query_feature = extract_feature(model,dataloaders['query_drone_160k'], view='drone', ms=multiple_scale, testing=True)
            gallery_feature = extract_feature(model,dataloaders['gallery_satellite_160k'], view='satellite', ms=multiple_scale)
        
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

    def get_competition_temp(self, data160k_dir='D://dataset/university-160k-wx', save_file = 'answer.txt', pth=None, multiple_scale=[1], batchsize=128):

        data_transforms = transforms.Compose([
            transforms.Resize((self.h, self.w), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image_datasets = {}
        image_datasets['gallery_satellite_160k'] = CustomData160k_sat(os.path.join(data160k_dir, 'gallery_satellite_160k'), data_transforms)

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                                        shuffle=False, num_workers=16) for x in
                        ['gallery_satellite_160k']}
        
        gallery_path = image_datasets['gallery_satellite_160k'].imgs
        gallery_label, gallery_path  = get_SatId_160k(gallery_path)
        
        # load model
        model = self.load_model(pth=pth)

        # Extract features
        with torch.no_grad():
            #query_feature = extract_feature(model,dataloaders['query_drone_160k'], view='drone', ms=multiple_scale, testing=True)
            gallery_feature = extract_feature(model,dataloaders['gallery_satellite_160k'], view='satellite', ms=multiple_scale)
        
        torch.save(gallery_feature, 'sat_'+pth)


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='MLPN')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true', help='get competition result')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--train_data_dir', default='/dataset/University-Release/train')
    parser.add_argument('--data160k_dir', default='/dataset/university-160k-wx')
    parser.add_argument('--pth', default='MLPN_170_')
    parser.add_argument('--num_epochs', default=180)
    parser.add_argument('--update_aug_img', default=[40,80,120,140,160,180,200])



    args = parser.parse_args()
    # update_aug_img = args.update_aug_img
    update_aug_img = list(range(10,args.num_epochs))

    m = MLPN_()
    if args.train:
        m.train(data_dir=args.train_data_dir, model_name='MLPN', num_epochs=220, checkpoint_interval=20, checkpoint_start=10, num_worker_imgaug=32, update_aug_img=update_aug_img)
    if args.test:
        m.get_competition_submit(data160k_dir=args.data160k_dir, pth=args.pth)
    if args.cache:
        m.get_competition_temp(data160k_dir=args.data160k_dir, pth=args.pth)
