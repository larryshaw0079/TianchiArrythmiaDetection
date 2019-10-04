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


class F1ScoreLoss(nn.Module):
    """
    F1 score as loss
    """
    def __init__(self, num_classes):
        super(F1ScoreLoss, self).__init__()

        self.num_classes = num_classes

    def forward(self, out, target):
        loss = 0
        for i in np.eye(self.num_classes):
            y_true = torch.from_numpy(i.reshape(1,-1))*target
            y_pred = torch.form_numpy(i.reshape(1,-1))*out
            loss += 0.5 * torch.sum(y_true * y_pred) / torch.sum(y_ture + y_pred + 1e-7)

        return -torch.log(loss+1e-7)

    
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


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
