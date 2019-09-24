import os

import numpy as np

import torch
import torch.nn as nn


def set_gpu(num_gpu=1, verbose=True):
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    gpu_memory = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu_ids = np.argsort(gpu_memory[::-1])
    os.system('rm tmp')
    assert(num_gpu <= len(gpu_ids))
    os.environ['VISIBLE_DEVICES']=','.join(map(str, gpu_ids[:num_gpu]))
    if verbose:
        print('Current GPU [%s], free memory: [%s] MB'%(os.environ['VISIBLE_DEVICES'], ','.join(map(str, np.sort(gpu_memory[::-1])[:num_gpu]))))


def get_gpu(num_gpu=1):
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    gpu_memory = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu_ids = np.argsort(gpu_memory[::-1])
    os.system('rm tmp')
    assert(num_gpu <= len(gpu_ids))

    return gpu_ids[:num_gpu]


class WeightedCrossEntropy(nn.Module):
    """
    The weighted cross entropy loss
    """
    def __init__(self, weights):
        super(WeightedCrossEntropy, self).__init__()
        self.base_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, out, target):
        loss = self.base_criterion(out, target)
        loss = (loss*self.weights).mean()

        return loss

    
class FocalLossMultiClass(nn.Module):
    """
    The focal loss for multi-class classification
    """
    def __init___(self, alpha, gamma):
        super(FocalLossMultiClass, self).__init__()
        self.__dict__.update(locals())
        # TODO

    def forward(self, out, target):
        pass # TODO
