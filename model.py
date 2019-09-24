import os
import pdb

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


class ECGData(Dataset):
    """
    The ECG Dataset class
    @param phase: Load training data of testing data
    @param extend: Compute the other 4 channels
    """
    def __init__(self, phase='train', extend=True):
        super(ECGData, self).__init__()
        assert(phase in ['train', 'test'])

        self.phase = phase
        self.extend = extend
        self.target = None
        if phase=='train':
            self.target = pd.read_csv('data/train/labels.csv')
        self.dirs = os.listdir('data/%s/raw_data'%phase)

    def __getitem__(self, idx):
        """
        Get the data item by an index
        """
        if self.phase=='train':
            # In the train phase, return the features (1, num_channels, time_length) and the target (num_classes,)
            x = np.loadtxt('data/train/raw_data/%d.txt'%(self.target.loc[idx, 'ID']), delimiter=' ', skiprows=1)
            if self.extend:
                extended_channels = np.zeros((x.shape[0], 4))
                extended_channels[:, 0] = x[:, 1] - x[:, 0]
                extended_channels[:, 1] = -(x[:, 1] + x[:, 0])/2
                extended_channels[:, 2] = x[:, 0] - x[:, 1]/2
                extended_channels[:, 3] = x[:, 1] - x[:, 0]/2
                x = np.concatenate([x, extended_channels], axis=1)
            x = torch.from_numpy(np.transpose(x).astype(np.float32)).cuda()
            y = torch.from_numpy(self.target.iloc[idx, 3:].values.astype(np.int)).cuda()
            return x, y
        else:
            # In the test phase, return the features (1, num_channels, time_length) only
            x = np.loadtxt('data/test/raw_data/%s'%(self.dirs[idx]), delimiter=' ', skiprows=1)
            x = torch.from_numpy(np.transpose(x.astype(np.float32))).cuda()
            return x

    def __len__(self):
        """
        Get the total number of instances
        """
        return len(self.dirs)

    def get_class_distribution(self):
        assert(self.phase == 'train')
        classes = self.target.iloc[:, 3:].values

        return np.sum(classes, axis=0)


class ResidualBlock(nn.Module):
    """
    The residual block
    @param input_channels: The number of channels of the input data
    @param output_channels: The number of channels of the output
    @param stride: The stride
    """
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

        # Only conv2 degrades the scale
        self.conv2 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(output_channels),
        )

        self.relu = nn.ReLU(inplace=True)


        # If stride == 1, the length of the time dimension will not be changed
        # If input_channels == output_channels, the number of channels will not be changed
        # If the channels are mismatch, the conv1d is used to upgrade the channel
        # If the time dimensions are mismatch, the conv1d is used to downsample the scale
        self.downsample = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm1d(output_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(x)
        out = self.conv3(x)

        residual = self.downsample(x) # Downsampe is an empty list if the size of inputs and outputs are same
        out += residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    The ResNet model
    @param input_channels: The number of channels of the input data
    @param hidden_channels: The channels of the output of the first conv1d layer
    @param num_classes: The number of classes of the target
    """
    def __init__(self, input_channels, hidden_channels, num_classes):
        super(ResNet, self).__init__()

        # The first convolution layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Residual layers
        self.layer1 = self.__make_layer(ResidualBlock, hidden_channels, hidden_channels, 2, stride=1)
        self.layer2 = self.__make_layer(ResidualBlock, hidden_channels, hidden_channels*2, 2, stride=2)
        self.layer3 = self.__make_layer(ResidualBlock, hidden_channels*2, hidden_channels*4, 2, stride=2)
        self.layer4 = self.__make_layer(ResidualBlock, hidden_channels*4, hidden_channels*8, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # Pooling operation computes the average of the last dimension (time dimension)

        # A dense layer for output
        self.fc = nn.Linear(hidden_channels*8, num_classes)

    def __make_layer(self, block, input_channels, output_channels, num_blocks, stride):
        """
        Get the residual layer
        @param block: The residual block
        @param input_channels: The number of input channels
        @param output_channels: The number of output channels
        @param num_blocks: The number of blocks in the layer
        @param stride: The stride of the convolution layer
        @return Torch.nn.Sequential
        """
        layers = []
        layers.append(block(input_channels, output_channels, stride=stride))
        for i in range(1, num_blocks):
            layers.append(block(output_channels, output_channels, stride=1))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        L_out = floor[(L_in + 2*padding - kernel) / stride + 1]
        @param x: (batch_size: 200, num_channels: 12, time_length: 5000)
        """
        out = self.conv1(x)           # (batch_size: 200, num_channels: 64, time_length: 2500)
        out = self.layer1(out)        # (batch_size: 200, num_channels: 64, time_length: 2500)
        out = self.layer2(out)        # (batch_size: 200, num_channels: 128, time_length: 1250)
        out = self.layer3(out)        # (batch_size: 200, num_channels: 256, time_length: 625)
        out = self.layer4(out)        # (batch_size: 200, num_channels: 512, time_length: 313)
        out = self.avg_pool(out)      # (batch_size: 200, num_channels: 512, time_length: 1)
        out = out.view(x.size(0), -1) # (batch_size: 200, num_channels: 512)
        out = self.fc(out)            # (batch_size: 200, num_channels: 55)

        return out
