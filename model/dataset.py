import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from config import *

class ECGData(Dataset):
    """
    The ECG Dataset class
    @param phase: Load training data of testing data
    @param extend: Compute the other 4 channels
    """
    def __init__(self, phase='train', extend=True):
        super(ECGData, self).__init__()
        assert(phase in ['train', 'val', 'test'])

        self.phase = phase
        self.extend = extend
        self.category_path = 'data/train/hf_round1_arrythmia.txt'
        self.train_label_path = 'data/train/hf_round1_label.txt'
        self.test_label_path = 'data/test/hf_round1_subA.txt'

        with open(self.category_path, 'r', encoding='utf8') as f:
            self.category = f.read().split('\n')[:-1]
        self.category2int = {name:idx for idx, name in enumerate(self.category)}

        if phase != 'test':
            with open(self.train_label_path, 'r', encoding='utf8') as f:
                contents = f.readlines()
                self.train_labels = {}
                for i, line in enumerate(contents):
                    line = line.split('\t')
                    self.train_labels[line[0]] = np.zeros(NUM_CLASSES)
                    for label in line[3:-1]:
                        self.train_labels[line[0]][self.category2int[label]] = 1

        if phase == 'test':
            with open(self.test_label_path, 'r', encoding='utf8') as f:
                contents = f.readlines()
                self.test_candidates = [line.split('\t')[0] for line in contents]

    def __getitem__(self, idx):
        """
        Get the data item by an index
        """
        if self.phase=='train':
            # In the train phase, return the features (1, num_channels, time_length) and the target (num_classes,)
            x = np.loadtxt('data/train/raw_data/%s'%(list(self.train_labels.keys())[idx]), delimiter=' ', skiprows=1)
            if self.extend:
                extended_channels = np.zeros((x.shape[0], 4))
                extended_channels[:, 0] = x[:, 1] - x[:, 0]
                extended_channels[:, 1] = -(x[:, 1] + x[:, 0])/2
                extended_channels[:, 2] = x[:, 0] - x[:, 1]/2
                extended_channels[:, 3] = x[:, 1] - x[:, 0]/2
                x = np.concatenate([x, extended_channels], axis=1)
            x = torch.from_numpy(np.transpose(x).astype(np.float32))
            y = torch.from_numpy(list(self.train_labels.values())[idx].astype(np.float32))
            return x, y
        else:
            # In the test phase, return the features (1, num_channels, time_length) only
            x = np.loadtxt('data/test/raw_data/%s'%(self.test_candidates[idx]), delimiter=' ', skiprows=1)
            if self.extend:
                extended_channels = np.zeros((x.shape[0], 4))
                extended_channels[:, 0] = x[:, 1] - x[:, 0]
                extended_channels[:, 1] = -(x[:, 1] + x[:, 0])/2
                extended_channels[:, 2] = x[:, 0] - x[:, 1]/2
                extended_channels[:, 3] = x[:, 1] - x[:, 0]/2
                x = np.concatenate([x, extended_channels], axis=1)
            x = torch.from_numpy(np.transpose(x.astype(np.float32)))
            return x

    def __len__(self):
        """
        Get the total number of instances
        """
        if self.phase == 'train':
            return len(self.train_labels)
        else:
            return len(self.test_candidates)

    def get_class_distribution(self):
        assert(self.phase == 'train')
        classes = np.array([item for item in self.train_labels.values()])

        return np.sum(classes, axis=0)