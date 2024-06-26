import tarfile
import zipfile 
from LPN.image_folder_ import CustomData160k_drone, CustomData160k_sat
import torch 
import numpy as np


def get_SatId_160k(img_path):
    labels = []
    paths = []
    for path,v in img_path:
        labels.append(v)
        paths.append(path)
    return labels, paths

def get_result_rank10(qf,gf,gl):
    query = qf.view(-1,1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    rank10_index = index[0:10]
    result_rank10 = gl[rank10_index]
    return result_rank10