import os

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import precision_score, recall_score, f1_score

from model import ECGData, ResNet
from util import set_gpu, WeightedCrossEntropy, FocalLossMultiClass
from config import *


def train(epoch, model, optimizer, criterion, train_loader, val_loader=None):
    total_train_loss = []
    total_val_loss = []
    with tqdm(train_loader, desc='Epoch: [%d/%d]'%(epoch+1, EPOCHS)) as loader:
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            total_train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            loader.set_postfix({'train_loss': np.mean(total_train_loss), 'val_loss': np.mean(total_val_loss)})

            if val_loader is not None and i % 10 == 0:
                for x, y in val_loader:
                    out = model(x)
                    loss = criterion(out, y)
                    totoal_val_loss.append(loss.item())
                    loader.set_postfix({'train_loss': np.mean(total_train_loss), 'val_loss': np.mean(total_val_loss)})


def test(model, test_loader):
    with torch.no_grad():
        output = np.array([model(x).detach().cpu().numpy() for x in test_loader])
    output = np.squeeze(output, axis=1)
    result = np.zeros_like(output).astype(np.int)
    result[output>=0.5] = 1

    with open('data/train/hf_round1_arrythmia.txt', 'r', encoding='utf8') as f:
        categories = f.read().strip().split('\n')

    with open('data/test/hf_round1_subA.txt', 'r', encoding='utf8') as f:
        test_contents = f.readlines()

    with open('testA.txt', 'w', encoding='utf8') as f:
        for i, line in enumerate(test_contents):
            line = line[:-1]
            for j in range(result.shape[1]):
                if result[i,j] == 1:
                    line+='%s\t'%(categories[j])
            line = line[:-1]+'\n'
            f.write(line)
    
    return result


if __name__ == '__main__':
    set_gpu(verbose=True)

    # Initialize the model and the optimizer
    model = ResNet(input_channels=INPUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, num_classes=NUM_CLASSES).cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(model)

    # Dataset
    dataset = ECGData(phase='train')
    weights = F.softmin(torch.tensor(dataset.get_class_distribution().astype(np.float32), requires_grad=False).cuda())
    train_size = int(len(dataset)*TRAIN_SPLIT)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Training
    criterion = WeightedCrossEntropy(weights)
    model.train()
    for epoch in range(EPOCHS):
        train(epoch, model, optimizer, criterion, train_loader, val_loader)

    # Testing
    test_dataset = ECGData(phase='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    result = test(model, test_loader)

