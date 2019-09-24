import numpy as np

import torch
import torch.nn as nn


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
