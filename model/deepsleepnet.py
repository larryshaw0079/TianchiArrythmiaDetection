import torch
import torch.nn as nn
import torch.functional as F

class CNN_Block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(CNN_Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class FeatureBlock(nn.Module):
    # batch_size, channels: 8(12 if extended), sequence_length: 5000
    def __init__(self, input_channels, hidden_channels, is_short):
        super(FeatureBlock, self).__init__()
        sample_rate = 500
        if is_short:
            # batch_size, channels: 64, sequence_length: 160
            self.conv1 = CNN_Block(input_channels, hidden_channels, kernel_size=251, stride=32, padding=125)
            # batch_size, channels: 64, sequence_length: 40
            self.max_pool1 = nn.MaxPool1d(kernel_size=9, stride=4, padding=4)
            # batch_size, channels: 128, sequence_length: 40
            self.layer_CNN = self.__make_layer(CNN_Block, hidden_channels, kernel_size=9, stride=1, padding=4, block_num=3)
            # batch_size, channels: 128, sequence_length: 10
            self.max_pool2 = nn.MaxPool1d(kernel_size=5, stride=4, padding=2)
        else:
            # batch_size, channels: 64, sequence_length: 40
            self.conv1 = CNN_Block(input_channels, hidden_channels, kernel_size=1001, stride=128, padding=500)
            # batch_size, channels: 64, sequence_length: 20
            self.max_pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
            # batch_size, channels: 128, sequence_length: 20
            self.layer_CNN = self.__make_layer(CNN_Block, hidden_channels, kernel_size=7, stride=1, padding=3, block_num=3)
            # batch_size, channels: 128, sequence_length: 10
            self.max_pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.drop_out = nn.Dropout(p=0.5, inplace=True)

    def __make_layer(self, block, hidden_channels, kernel_size, stride, padding, block_num):
        layer = []
        layer.append(block(hidden_channels, 2*hidden_channels, kernel_size, stride, padding))
        for i in range(1, block_num):
            layer.append(block(hidden_channels*2, hidden_channels*2, kernel_size, stride, padding))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)

        out = self.max_pool1(out)

        out = self.drop_out(out)

        out = self.layer_CNN(out)

        out = self.max_pool2(out)
        return out

class BiLSTMBlock(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers):
        super(BiLSTMBlock,  self).__init__()
        self.bi_lstm = nn.Sequential(
            nn.LSTM(input_channels, hidden_channels, num_layers, batch_first=True, bidirectional=True),
            nn.Dropout(p=0.5, inplace=True)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()

        out, _ = self.bi_lstm(x, (h0, c0)) # batch_size, seq_length, hidden_size*2
        return out

class DeepSleepNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes, sample_rate=500):
        super(DeepSleepNet, self).__init__()
        self.feature_short = FeatureBlock(input_channels, hidden_channels=hidden_channels, is_short=True)
        self.feature_long = FeatureBlock(input_channels, hidden_channels=hidden_channels, is_short=False)

        self.dropout = nn.Dropout(p=0.5, inplace=True)

        self.bi_lstm = nn.Sequential(
            BiLSTMBlock(4*hidden_channels, 2*hidden_channels, 1),
            BiLSTMBlock(4*hidden_channels, 2*hidden_channels, 1),
        )

        self.shortcut = nn.Linear(4*hidden_channels, 4*hidden_channels)
        self.fc =  nn.Linear(4*hidden_channels, num_classes)

    def forward(self, x):
        sf = self.feature_short(x)      # batch_size, channels: 128, sequence_length: 10
        lf = self.feature_long(x)       # batch_size, channels: 128, sequence_length: 10
        out = torch.cat((sf, lf), 1)    # batch_size, channels: 256, sequence_length: 10

        out = self.dropout(out)

        shortcut = self.shortcut(out)

        out = out.transpose(1, 2)
        out = self.bi_lstm(out)[:, -1, :]

        out = self.dropout(out+shortcut)
        out = self.fc(out)
        return out