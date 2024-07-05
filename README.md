# Code for UAVM'24 Challenge

code of Team liyiqing_cs on [ACMMM24 Multimedia Drone Satellite Matching Challenge In Multiple-environment](https://codalab.lisn.upsaclay.fr/competitions/18770#results)

code waits to be cleaned

## Idea list

### Frequency Distenglement

motivated by [Generalized UAV Object Detection via Frequency Domain Disentanglement (CVPR 23)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Generalized_UAV_Object_Detection_via_Frequency_Domain_Disentanglement_CVPR_2023_paper.pdf) , adopt Frequency Distenglement in drone-view localization

- `Ours*`: train a weather classification network; jointly optimize frequency domain filter and MLPN for domain generaliztion (only drone-view image go through filter, and exists two filter that extract domain invariant spectrum and domain specific spectrum)

- `ffm_test.py/ffm_train.py`: jointly optimize frequency domain filter and MLPN for domain generaliztion (only drone-view image go through filter, and exists only one filter that extract domain invariant spectrum)
    - `ffm_train_iter`: use alternating optimiztion in `ffm_train` 

- `ffm2.py`: jointly optimize frequency domain filter and MLPN for domain generaliztion ( both satllite view image and drone-view image go through filter)

### Image2Image Module

train Image2Image Module to change multi-weather domain image into normal image

- `ffm_naive`: train frequency filter as Image2Image Module
- `repair*`: train a toy GAN as Image2Image Module (not working)
- `simple_replace.ipynb`: as a part to use [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) as Image2Image Module (training stage code is not included in this repo)

**HOWEVER, we get AP@1 85.08 by using none of ideas above, just by simply using MLPN with augmented University-1652's training set**
## Baselines

We included the code of 

- LPN 
    - `./LPN`
    - `LPN.ipynb`: test LPN on university-1652 and a mixed scenerio
    - `LPN-wx.ipynb` test LPN on augmented 

- MuseNet
    - `./MuseNet`
    - `MuseNet.ipynb`: test MuseNet on mixed scenerio

- MLPN
    - `./MLPN`
    - `MLPN.ipynb`: test MLPN on mixed scenerio
    - `MLPN_ensemble.ipynb`: train 10 MLPN for each weather case and ensemble them

as baselines and run the related experiments

## utils

- `./utils`

some common code for all these methods.

