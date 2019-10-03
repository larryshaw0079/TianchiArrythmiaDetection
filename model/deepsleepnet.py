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
        """
        feature extraction block, using cnn with different kernel size to extract both fine-grained & coerse-grained feature
        :param input_channels: input data channels, default 8 (12 if extended)
        :param hidden_channels:
        :param is_short: flag to determined the kernel size for either fine-grained or coerse-grained features
        """
        super(FeatureBlock, self).__init__()
        # data sample rate, 500 hz, 10s ECG data
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
        """
        BiDirectional LSTM Block, for learning tempory dependency
        :param input_channels:
        :param hidden_channels:
        :param num_layers:
        """
        super(BiLSTMBlock, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.bi_lstm = nn.LSTM(input_channels, hidden_channels, num_layers, batch_first=True, bidirectional=True, dropout=0.5)


    def forward(self, x):
        h0 = torch.rand(self.num_layers*2, x.size(0), self.hidden_channels).cuda()
        c0 = torch.rand(self.num_layers*2, x.size(0), self.hidden_channels).cuda()
        out, _ = self.bi_lstm(x, (h0, c0)) # batch_size, seq_length, hidden_size*2
        return out

class DeepSleepNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes, sample_rate=500):
        """
        DeepSleepNet, basic intuition is to use both small & large kernel cnn to extract both fine-grained feature and coersive one
        by concatening those extracted feature as input feature to Bi-directional LSTM Module, this model could
        extract the temporial dependency for time-sequence data like ECG
        :param input_channels:
        :param hidden_channels:
        :param num_classes:
        :param sample_rate:
        """
        super(DeepSleepNet, self).__init__()
        self.feature_short = FeatureBlock(input_channels, hidden_channels=hidden_channels, is_short=True)
        self.feature_long = FeatureBlock(input_channels, hidden_channels=hidden_channels, is_short=False)

        self.dropout = nn.Dropout(p=0.5, inplace=True)

        self.bi_lstm = BiLSTMBlock(4*hidden_channels, 2*hidden_channels, 2)

        self.shortcut = nn.Linear(4*hidden_channels, 4*hidden_channels)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc =  nn.Linear(8*hidden_channels, num_classes)

    def forward(self, x):
        sf = self.feature_short(x)          # batch_size, channels: 128, sequence_length: 10
        lf = self.feature_long(x)           # batch_size, channels: 128, sequence_length: 10
        out = torch.cat((sf, lf), 1)        # batch_size, channels: 256, sequence_length: 10

        out = self.dropout(out)
        out = out.transpose(1, 2)
        shortcut = self.shortcut(out)       # batch_size, sequence_length: 10, channels: 256

        out = self.bi_lstm(out)             # batch_size, sequence_length: 10, channels: 256
        out = torch.cat((out, shortcut), -1)    # batch_size, sequence_length: 10, channels: 512
        out = self.dropout(out)

        out = self.avg_pool(out.transpose(1, 2))    #batch_size, channels: 512 sequence_length: 1
        out = out.view(x.size(0), -1)               #batch_size, channels: 512

        out = self.fc(out)                          #batch_size, channel:  55
        return out