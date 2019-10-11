import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    def __init__(self, input_sets):
        super(CNN, self).__init__()
        cnn = nn.Sequential()
        self.input_sets = input_sets
        imgH = 64
        nc = 3
        self.CNN_embed_dim = 300
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d(2, 2))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d(2, 2))  # 512*3*3
        convRelu(6, True)  # 512x1x16

        self.fc_hidden1 = 256
        self.fc_hidden2 = 256
        self.CNN_embed_dim = 300
        self.fc1 = nn.Linear(512*3*3, self.fc_hidden1)   # fully connected layer, output k classes
        input_length = 0
        for k, v in self.input_sets.items():
            setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
            input_length += 100

        self.fc2 = nn.Linear(self.fc_hidden1 + input_length, self.CNN_embed_dim)
        self.drop_p = 0.0
        self.cnn = cnn

    def forward(self, inputs):
        x_3d = inputs['image']
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.cnn(x_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)           # flatten the output of conv

            # FC layers
            x = F.relu(self.fc1(x))
            for k, v in self.input_sets.items():
                extra_input = getattr(self,'{}_input'.format(k))(inputs[k][:, t].to(torch.int64))
                x = torch.cat((x, extra_input), 1)
            x = F.relu(self.fc2(x))
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.GRU(nIn, nHidden, bidirectional=True, batch_first=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        b, T, h = recurrent.size()
        t_rec = recurrent.contiguous().view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(b, T, -1)

        return output


class StatusGRU(nn.Module):
    def __init__(self, sets, input_sets):
        super(StatusGRU, self).__init__()
        self.cnn = CNN(input_sets)

        self.input_dim = self.cnn.CNN_embed_dim
        self.hidden_dim = 128
        self.batch_size = 1
        self.h_RNN_layers = 3   # RNN hidden layers
        self.h_RNN = 256                 # RNN hidden nodes
        self.h_FC_dim = 128
        self.num_classes = len(sets['hero'])
        self.rnn = nn.Sequential(
            BidirectionalLSTM(self.input_dim, self.h_RNN, self.h_RNN),
            BidirectionalLSTM(self.h_RNN, self.h_RNN, self.h_RNN))
        self.sets = sets

        self.input_sets = input_sets
        self.embeddings = {}
        input_length = 0
        for k, v in self.input_sets.items():
            setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
            input_length += 100
        self.fc1 = nn.Linear(self.h_RNN, self.h_RNN)
        #self.fc2 = nn.Linear(256 + input_length, self.CNN_embed_dim)
        #self.fc1 = nn.Linear(self.h_RNN * 2, self.num_classes)
        #self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
        for k, v in self.sets.items():
            m = nn.Linear(self.h_RNN, len(v))
            #m = TimeDistributed(fully_connected)
            setattr(self, '{}_output'.format(k), m)
        self.drop_p = 0.0

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda())

    def forward(self, inputs):

        x_RNN = self.cnn(inputs)
        x = self.rnn(x_RNN)
        x = self.fc1(x)   # choose RNN_out at the last time step
        x = F.relu(x)
        #x = F.dropout(x, p=self.drop_p, training=self.training)
        #x = self.fc2(x)
        x_outs = {}
        for k in self.sets.keys():
            o = F.log_softmax(getattr(self,'{}_output'.format(k))(x), dim=2)
            x_outs[k] = o

        return x_outs
