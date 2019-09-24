import os

import numpy as np

import torch
import torch.nn as nn


def set_gpu(verbose=True):
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    gpu_memory = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu_id = np.argmax(gpu_memory)
    os.system('rm tmp')
    os.environ['VISIBLE_DEVICES']=str(gpu_id)
    if verbose:
        print('Current GPU [%d], free memory: %.0f MB'%(gpu_id, gpu_memory[gpu_id]))


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
