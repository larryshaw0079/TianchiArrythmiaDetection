import pdb

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """
    The inception block
    @param input_channels: 
    @param bottleneck_size:
    @param hidden_channels:
    @param max_kernel_size: 
    @param stride: 
    """
    def __init__(self, input_channels, bottleneck_size, hidden_channels, stride):
        super(InceptionBlock, self).__init__()
        self.__dict__.update(locals())

        self.bottleneck = nn.Sequential(
            nn.Conv1d(input_channels, bottleneck_size, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(bottleneck_size, hidden_channels, kernel_size=11, padding=5, stride=stride, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(bottleneck_size, hidden_channels, kernel_size=21, padding=10, stride=stride, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(bottleneck_size, hidden_channels, kernel_size=41, padding=20, stride=stride, bias=False),
            nn.ReLU(inplace=True)
        )

        self.max_pooling = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
        self.short_conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(
            nn.BatchNorm1d(hidden_channels*4),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x):
        bottleneck_x = self.bottleneck(x)
        out1 = self.conv1(bottleneck_x)
        out2 = self.conv2(bottleneck_x)
        out3 = self.conv3(bottleneck_x)
        
        pool_x = self.max_pooling(x)
        out4 = self.short_conv(pool_x)

        # Concatenate outputs on the channel dimension
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.output(out)

        return out


class InceptionTimeNet(nn.Module):
    """

    """
    def __init__(self, input_channels, bottleneck_size, hidden_channels, stride, num_classes):
        super(InceptionTimeNet, self).__init__()
        self.__dict__.update(locals())

        self.inception1 = InceptionBlock(input_channels=input_channels, bottleneck_size=bottleneck_size, hidden_channels=hidden_channels, stride=stride)
        self.inception2 = InceptionBlock(input_channels=hidden_channels*4, bottleneck_size=bottleneck_size, hidden_channels=hidden_channels, stride=stride)
        self.inception3 = InceptionBlock(input_channels=hidden_channels*4, bottleneck_size=bottleneck_size, hidden_channels=hidden_channels, stride=stride)
        self.inception4 = InceptionBlock(input_channels=hidden_channels*4, bottleneck_size=bottleneck_size, hidden_channels=hidden_channels, stride=stride)
        self.inception5 = InceptionBlock(input_channels=hidden_channels*4, bottleneck_size=bottleneck_size, hidden_channels=hidden_channels, stride=stride)
        self.inception6 = InceptionBlock(input_channels=hidden_channels*4, bottleneck_size=bottleneck_size, hidden_channels=hidden_channels, stride=stride)

        # Residual layer between the input layer and inception3
        self.shortcut1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels*4, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(hidden_channels*4)
        )
        # Residual layer between the inception3 and inception6
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(hidden_channels*4, hidden_channels*4, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(hidden_channels*4)
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1) # Pooling operation computes the average of the last dimension (time dimension)
        # A dense layer for output
        self.fc = nn.Linear(hidden_channels*4, num_classes)

    def forward(self, x):
        residual = x
        out = self.inception1(x)
        out = self.inception2(out)
        out = self.inception3(out)

        out += self.shortcut1(residual)
        out = F.relu(out, inplace=True)
        residual = out

        out = self.inception4(out)
        out = self.inception5(out)
        out = self.inception6(out)

        out += self.shortcut2(residual)
        out = F.relu(out, inplace=True)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out