import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


class ECGData(Dataset):
    def __init__(self, phase='train'):
        super(ECGData, self).__init__()
        assert(phase in ['train', 'test'])

        self.phase = phase
        self.target = None
        if phase=='train':
            self.target = pd.read_csv('data/train/labels.csv')
        self.dirs = os.listdirs('data/%s/raw_data'%phase)

    def __getitem__(self, idx):
        if phase=='train':
            x = np.loadtxt('data/train/raw_data/%d.txt'%(self.target.loc[idx, 'ID']), delimiter=' ', skiprows=1)
            return torch.from_numpy(x.astype(np.float32)).cuda(), torch.from_numpy(self.target.iloc[idx, 3:].values).cuda()
        else:
            x = np.loadtxt('data/test/raw_data/%s'%(self.dirs[idx]), delimiter=' ', skiprows=1)
            return torch.from_numpy(x.astype(np.float32)).cuda()

    def __len__(self):
        return len(self.dirs)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(output_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=stride, bias=False),
                nn.BatchNorm1d(output_channels)
            )

    def forward(self, x):
        out = self.layers(x)
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.__make_layer(ResidualBlock, hidden_channels, hidden_channels, 2, stride=1)
        self.layer2 = self.__make_layer(ResidualBlock, hidden_channels, hidden_channels*2, 2, stride=2)
        self.layer3 = self.__make_layer(ResidualBlock, hidden_channels*2, hidden_channels*4, 2, stride=2)
        self.layer4 = self.__make_layer(ResidualBlock, hidden_channels*4, hidden_channels*8, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels*8, num_classes)

    def __make_layer(self, block, input_channels, output_channels, num_blocks, stride):
        strides = [1] * (num_blocks-1)
        layers = []
        layers.append(block(input_channels, output_channels, stride=stride))
        for stride in strides:
            layers.append(block(output_channels, output_channels, stride=stride))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.vie(x.size(0), -1)
        out = self.fc(out)

        return out
