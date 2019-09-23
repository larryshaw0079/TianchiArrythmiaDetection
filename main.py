import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from model import ECGData, ResNet
from util import WeightedCrossEntropy
from config import *


model = ResNet(input_channels=INPUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, num_classes=NUM_CLASSES).cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train(epochs):

