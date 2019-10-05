import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class F1ScoreLoss(nn.Module):
    """
    F1 score as loss
    """
    def __init__(self, num_classes):
        super(F1ScoreLoss, self).__init__()

        self.num_classes = num_classes

    def forward(self, out, target):
        y_true = target
        y_pred = F.sigmoid(out)
        tp = torch.sum(y_true*y_pred)
        tn = torch.sum((1-y_true)*(1-y_pred))
        fp = torch.sum((1-y_true)*y_pred)
        fn = torch.sum(y_true*(1-y_pred))
        p = tp/(tp+fp+1e-7)
        r = tp/(tp+fn+1e-7)
        f1 = 2*p*r/(p+r+1e-7)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)

        return 1-torch.mean(f1)

    
class FocalLossMultiClass(nn.Module):
    """
    The focal loss for multi-class classification
    @param gamma: 
    @param weights: 
    """
    def __init___(self, gamma, weights):
        super(FocalLossMultiClass, self).__init__()
        self.__dict__.update(locals())

    def forward(self, out, target):
        y_true = target
        y_pred = F.sigmoid(out)

        loss = -((y_true*((1-y_pred)**self.gamma)*torch.log(y_pred) + (1-y_true)*(y_pred**self.gamma)*torch.log(1-y_pred)))
        loss = (loss*self.weights).mean()

        return loss


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
