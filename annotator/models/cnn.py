import torch
import torch.nn as nn
import torch.nn.functional as F

class StatusCNN(nn.Module):
    def __init__(self, sets, input_sets):
        super(StatusCNN, self).__init__()
        self.sets = sets
        self.input_sets = input_sets
        self.embeddings = {}
        cnn = nn.Sequential()
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

        self.cnn = cnn
        input_length = 0
        self.fc1 = nn.Linear(512*3*3, 256)
        for k, v in self.input_sets.items():
            setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
            input_length += 100
        self.fc2 = nn.Linear(256 + input_length, self.CNN_embed_dim)
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        inputs = {}
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        # conv features
        x = self.cnn(x)
        x = x.view(-1, 512*3*3)

        x = F.relu(self.fc1(x))
        for k, v in inputs.items():
            extra_input = getattr(self,'{}_input'.format(k))(v.to(torch.int64))
            x = torch.cat((x, extra_input), 1)
        x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs


class StatusCNNBackup(nn.Module):
    def __init__(self, sets, input_sets):
        super(StatusCNN, self).__init__()
        self.CNN_embed_dim = 300
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5)
        self.fc1 = nn.Linear(10 * 13 * 13, 128)
        self.sets = sets
        self.input_sets = input_sets
        self.embeddings = {}
        input_length = 0
        for k, v in self.input_sets.items():
            setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
            input_length += 100
        self.fc2 = nn.Linear(128 + input_length, self.CNN_embed_dim)
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        inputs = {}
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 13 * 13)

        x = F.relu(self.fc1(x))
        for k, v in inputs.items():
            extra_input = getattr(self,'{}_input'.format(k))(v.to(torch.int64))
            x = torch.cat((x, extra_input), 1)
        x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs

class MidCNN(nn.Module):
    def __init__(self, sets):
        super(MidCNN, self).__init__()
        cnn = nn.Sequential()
        imgH = 48
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

        self.cnn = cnn

        self.fc1 = nn.Linear(512*2*8, self.CNN_embed_dim)
        self.sets = sets
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        x = x['image']
        x = self.cnn(x)
        x = x.view(-1, 512*2*8)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs

class GameCNN(nn.Module):
    def __init__(self, sets):
        super(GameCNN, self).__init__()
        cnn = nn.Sequential()
        imgH = 144
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

        self.cnn = cnn

        self.fc1 = nn.Linear(512*8*15, self.CNN_embed_dim)
        self.sets = sets
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        x = x['image']
        x = self.cnn(x)
        x = x.view(-1, 512*8*15)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs

class PauseCNN(nn.Module):
    def __init__(self, sets):
        super(PauseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5)
        self.fc1 = nn.Linear(10 * 3 * 15, 128)
        self.sets = sets
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(128, len(v)))

    def forward(self, x):
        x = x['image']
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 3 * 15)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs

class ReplayCNN(nn.Module):
    def __init__(self, sets):
        super(ReplayCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5)
        self.fc1 = nn.Linear(10 * 5 * 23, 128)
        self.sets = sets
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(128, len(v)))

    def forward(self, x):
        x = x['image']
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 23)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs

class KillFeedCNN(nn.Module):
    def __init__(self, sets):
        super(KillFeedCNN, self).__init__()
        cnn = nn.Sequential()
        imgH = 32
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
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

        self.fc1 = nn.Linear(512*63, self.CNN_embed_dim)
        self.sets = sets
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        x = x['image']
        x = self.cnn(x)
        x = x.view(-1, 512*63)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs
