import os
from datetime import datetime

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split

import inspect
from gpu_mem_track import MemTracker
from torchsummary import summary

from sklearn.metrics import precision_score, recall_score, f1_score

from model import ECGData, ResNet
from util import WeightedCrossEntropy, FocalLossMultiClass
from config import *


def train(epoch, model, optimizer, criterion, train_loader, val_loader=None, writer=None):
    """
    Train the network for an epoch. Executing model.train() before invoking this function is recommended.
    @param epoch: The current epoch
    @param model: The current model
    @param optimizer: The current optimizer
    @param criterion: The loss function
    @param train_loader: Training dataset loader
    @param val_loader: Validation dataset loader
    @param writer: The tensorboard writer
    """
    total_train_loss = []
    total_val_loss = []
    with tqdm(train_loader, desc='Epoch: [%d/%d]'%(epoch+1, EPOCHS)) as loader:
        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            total_train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            loader.set_postfix({'train_loss': np.nan if len(total_train_loss)==0 else np.mean(total_train_loss), 'val_loss': np.nan if len(total_val_loss)==0 else np.mean(total_val_loss)})

            if val_loader is not None and (i+1) % 100 == 0:
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.cuda(), y.cuda()
                        out = model(x)
                        loss = criterion(out, y)
                        total_val_loss.append(loss.item())
                        loader.set_postfix({'train_loss': np.nan if len(total_train_loss)==0 else np.mean(total_train_loss), 'val_loss': np.nan if len(total_val_loss)==0 else np.mean(total_val_loss)})
    
    if writer is not None:
        writer.add_scalar('Loss/train', np.mean(total_train_loss), epoch)
        writer.add_scalar('Loss/val', np.mean(total_val_loss), epoch)


def evaluate(model, data_loader):
    """
    Evaluate the model performance. Executing model.eval() before invoking this function is recommended.
    """
    results = []
    target = []
    with torch.no_grad():
        for x, y in tqdm(data_loader):
            x = x.cuda()
            output = F.sigmoid(model(x)).detach().cpu().numpy()
            results.append(output)
            target.append(y.numpy())
    result_expect_last = np.array(results[:-1]).reshape(-1, NUM_CLASSES)
    result_last = np.array(results[-1]).reshape(-1, NUM_CLASSES)
    results = np.concatenate((result_expect_last, result_last), axis=0)

    target_expect_last = np.array(target[:-1]).reshape(-1, NUM_CLASSES)
    target_last = np.array(target[-1]).reshape(-1, NUM_CLASSES)
    target = np.concatenate((target_expect_last, target_last), axis=0).astype(np.int)

    result = np.zeros_like(results).astype(np.int)
    result[results>=0.5] = 1

    precision_micro = precision_score(result, target, average='micro')
    precision_macro = precision_score(result, target, average='macro')
    recall_micro = recall_score(result, target, average='micro')
    recall_macro = recall_score(result, target, average='macro')
    f1_micro = f1_score(result, target, average='micro')
    f1_macro = f1_score(result, target, average='macro')

    with open('output/result-%s.txt'%(str(datetime.now()).strip().replace(':','-')), 'w', encoding='utf8') as f:
        f.write('Presion Micro: %f\n'%precision_micro)
        f.write('Presion Macro: %f\n'%precision_macro)
        f.write('Recall Micro: %f\n'%recall_micro)
        f.write('Recall Macro: %f\n'%recall_macro)
        f.write('F1 Micro: %f\n'%f1_micro)
        f.write('F1 Macro: %f\n'%f1_macro)


def test(model, test_loader):
    """
    Output results. Executing model.eval() before invoking this function is recommended.
    """
    with torch.no_grad():
        output = [F.sigmoid(model(x.cuda())).detach().cpu().numpy() for x in tqdm(test_loader)]
        output_expect_last = np.array(output[:-1]).reshape(-1, NUM_CLASSES)
        output_last = np.array(output[-1]).reshape(-1, NUM_CLASSES)
        output = np.concatenate((output_expect_last, output_last), axis=0)
    result = np.zeros_like(output).astype(np.int)
    result[output>=0.5] = 1

    with open('data/train/hf_round1_arrythmia.txt', 'r', encoding='utf8') as f:
        categories = f.read().strip().split('\n')

    with open('data/test/hf_round1_subA.txt', 'r', encoding='utf8') as f:
        test_contents = f.readlines()

    with open('output/testA-%s.txt'%(str(datetime.now()).strip().replace(':','-')), 'w', encoding='utf8') as f:
        for i, line in tqdm(enumerate(test_contents)):
            line = line[:-1]
            for j in range(result.shape[1]):
                if result[i,j] == 1:
                    line+='%s\t'%(categories[j])
            line = line[:-1]+'\n'
            f.write(line)
    
    return result


def save_model(model, path):
    torch.save(model, path)


def save_parameters(model, path):
    torch.save(model.state_dict(), path)


def load_model(path):
    model = torch.load(path)
    
    return model


def load_parameters(model, path):
    model.load_state_dict(torch.load(path))

    return model


if __name__ == '__main__':
    if MULTI_GPU:
        num_gpu = 3
    else:
        num_gpu = 1
    verbose = True
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    gpu_memory = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    gpu_ids = np.argsort(-1*np.array(gpu_memory))
    os.system('rm tmp')
    assert(num_gpu <= len(gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str, gpu_ids[:num_gpu]))
    if verbose:
        print('Current GPU [%s], free memory: [%s] MB'%(os.environ['CUDA_VISIBLE_DEVICES'], ','.join(map(str, np.array(gpu_memory)[gpu_ids[:num_gpu]]))))

    frame = inspect.currentframe() # define a frame to track
    gpu_tracker = MemTracker(frame, path='log/') # define a GPU tracker
    gpu_tracker.track() # run function between the code line where uses GPU

    if MULTI_GPU:
        # Initialize the model and the optimizer        
        model = ResNet(input_channels=INPUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, num_classes=NUM_CLASSES, dilated=DILATED)
        model = nn.DataParallel(model).cuda()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        # Initialize the model and the optimizer
        model = ResNet(input_channels=INPUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, num_classes=NUM_CLASSES, dilated=DILATED).cuda()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    summary(model, (INPUT_CHANNELS, 5000))
    gpu_tracker.track() # run function between the code line where uses GPU

    writer = None
    if ENABLE_TENSORBOARD:
        writer = SummaryWriter(logdir='log/')
        # add_graph on multi-gpu is not support
        if not MULTI_GPU:
            dummy_input = torch.rand(2, INPUT_CHANNELS, 5000, requires_grad=False).cuda()
            writer.add_graph(model, (dummy_input, ))

    # Dataset
    dataset = ECGData(phase='train')
    # weights = F.softmin(torch.tensor(dataset.get_class_distribution().astype(np.float32), requires_grad=False).cuda())
    weights = torch.tensor(1/np.log(dataset.get_class_distribution().astype(np.float32)), requires_grad=False).cuda()
    train_size = int(len(dataset)*TRAIN_SPLIT)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Training
    print('==================================================')
    print('| Trainging stage started.')
    print('==================================================')
    criterion = WeightedCrossEntropy(weights)
    model.train()
    for epoch in range(EPOCHS):
        train(epoch, model, optimizer, criterion, train_loader, val_loader, writer)
    gpu_tracker.track() # run function between the code line where uses GPU

    if SAVE_MODEL:
        save_model(model, path='output/%s_epoch%d_%s.pkl'%(SAVE_NAME, EPOCHS, str(datetime.now())))

    # Evaluating
    print('==================================================')
    print('| Evaluating stage started.')
    print('==================================================')
    model.eval()
    evaluate(model, val_loader)

    # Testing
    print('==================================================')
    print('| Testing stage started.')
    print('==================================================')
    test_dataset = ECGData(phase='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    model.eval()
    result = test(model, test_loader)

    writer.close()
