import torch
import torch.nn as nn
import torch.nn.functional as F


class StatusCNN(nn.Module):
    def __init__(self, sets, input_sets):
        super(StatusCNN, self).__init__()
        self.sets = sets
        self.input_sets = input_sets
        self.drop_p = 0.3
        cnn = nn.Sequential()
        nc = 3
        self.CNN_embed_dim = 300
        self.bottleneck_dim = 256
        #assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
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
        self.bottleneck = nn.Linear(512*3*3, self.bottleneck_dim)
        input_length = 6
        if self.input_sets:
            for k, v in self.input_sets.items():
                setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
                input_length += 100
        self.classifier = nn.Sequential(
            nn.Linear(self.bottleneck_dim + input_length, self.CNN_embed_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_p)
        )
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        color = x['color']
        enemy_color = x['enemy_color']
        inputs = {}
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        # conv features
        x = self.cnn(x)
        x = x.view(-1, 512*3*3)
        x = F.relu(self.bottleneck(x))
        if self.input_sets:
            for k, v in inputs.items():
                extra_input = getattr(self,'{}_input'.format(k))(v.to(torch.int64))
                x = torch.cat((x, extra_input), 1)
        x = torch.cat((x, color), 1)
        x = torch.cat((x, enemy_color), 1)

        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.classifier(x)
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = F.log_softmax(getattr(self,'{}_output'.format(k))(x), dim=1)
        return x_outs


class StatusTestCNN(nn.Module):
    def __init__(self, sets, input_sets):
        super(StatusTestCNN, self).__init__()
        self.sets = sets
        self.input_sets = input_sets
        self.drop_p = 0.3
        cnn = nn.Sequential()
        nc = 3
        self.CNN_embed_dim = 300
        #assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
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
        input_length += 6
        self.fc2 = nn.Linear(256 + input_length, self.CNN_embed_dim)
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        color = x['color']
        enemy_color = x['enemy_color']
        inputs = {}
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        # conv features
        x = self.cnn(x)
        x = x.view(-1, 512*3*3)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = F.relu(self.fc1(x))
        for k, v in inputs.items():
            extra_input = getattr(self,'{}_input'.format(k))(v.to(torch.int64))
            x = torch.cat((x, extra_input), 1)
        x = torch.cat((x, color), 1)
        x = torch.cat((x, enemy_color), 1)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = F.log_softmax(getattr(self,'{}_output'.format(k))(x), dim=1)
        return x_outs

class StatusNormCNN(nn.Module):
    def __init__(self, sets, input_sets):
        super(StatusCNN, self).__init__()
        self.sets = sets
        self.input_sets = input_sets
        self.drop_p = 0.3
        cnn = nn.Sequential()
        nc = 3
        self.CNN_embed_dim = 300
        #assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        def convRelu(i, batchNormalization=True):
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
        self.bn1 = nn.BatchNorm1d(256)
        for k, v in self.input_sets.items():
            setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
            input_length += 100
        input_length += 6
        self.fc2 = nn.Linear(256 + input_length, self.CNN_embed_dim)
        self.bn2 = nn.BatchNorm1d(self.CNN_embed_dim)
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        color = x['color']
        enemy_color = x['enemy_color']
        inputs = {}
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        # conv features
        x = self.cnn(x)
        x = x.view(-1, 512*3*3)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        for k, v in inputs.items():
            extra_input = getattr(self,'{}_input'.format(k))(v.to(torch.int64))
            x = torch.cat((x, extra_input), 1)
        x = torch.cat((x, color), 1)
        x = torch.cat((x, enemy_color), 1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = F.log_softmax(getattr(self,'{}_output'.format(k))(x), dim=1)
        return x_outs


class WindowedStatusCNN(nn.Module):
    def __init__(self, sets, input_sets):
        super(WindowedStatusCNN, self).__init__()
        self.sets = sets
        self.input_sets = input_sets
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
        self.fc1 = nn.Linear(3*512*3*3, 256)
        for k, v in self.input_sets.items():
            setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
            input_length += 100
        input_length += 6
        self.fc2 = nn.Linear(256 + input_length, self.CNN_embed_dim)
        self.drop_p = 0.3
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        inputs = {}
        color = x['color']
        enemy_color = x['enemy_color']
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        batch_size, num_steps, num_channels, height, width = x.size()
        x = x.view(batch_size*num_steps, num_channels, height, width)
        # conv features
        x = self.cnn(x)
        x = x.view(batch_size, num_steps * 512*3*3)
        #x = x.view(-1, 512*3*3)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = F.relu(self.fc1(x))
        for k, v in inputs.items():
            extra_input = getattr(self,'{}_input'.format(k))(v.to(torch.int64))
            x = torch.cat((x, extra_input), 1)
        x = torch.cat((x, color), 1)
        x = torch.cat((x, enemy_color), 1)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = F.log_softmax(getattr(self,'{}_output'.format(k))(x), dim=1)
        return x_outs


class MidCNN(nn.Module):
    def __init__(self, sets, input_sets):
        super(MidCNN, self).__init__()
        cnn = nn.Sequential()
        self.sets = sets
        imgH = 48
        nc = 3
        self.drop_p = 0.3
        self.CNN_embed_dim = 300
        self.bottleneck_dim = 256
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

        self.bottleneck = nn.Linear(512*2*8, self.bottleneck_dim)
        self.input_sets = input_sets
        self.embeddings = {}
        input_length = 0
        for k, v in self.input_sets.items():
            setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
            input_length += 100

        self.classifier = nn.Sequential(
            nn.Linear(self.bottleneck_dim + input_length, self.CNN_embed_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_p)
        )
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        inputs = {}
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        x = self.cnn(x)
        x = x.view(-1, 512*2*8)
        x = F.relu(self.bottleneck(x))
        if self.input_sets:
            for k, v in inputs.items():
                extra_input = getattr(self,'{}_input'.format(k))(v.to(torch.int64))
                x = torch.cat((x, extra_input), 1)

        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.classifier(x)
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = F.log_softmax(getattr(self,'{}_output'.format(k))(x), 1)
        return x_outs


class GameCNN(nn.Module):
    def __init__(self, sets, input_sets):
        super(GameCNN, self).__init__()
        cnn = nn.Sequential()
        self.sets = sets
        self.input_sets = input_sets
        self.drop_p = 0.3
        imgH = 144
        nc = 3
        self.CNN_embed_dim = 300
        self.bottleneck_dim = 256
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        def convRelu(i, batchNormalization=True):
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
        self.bottleneck = nn.Linear(512*8*15, self.bottleneck_dim)
        input_length = 0
        if self.input_sets:
            for k, v in self.input_sets.items():
                setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
                input_length += 100

        self.classifier = nn.Sequential(
            nn.Linear(self.bottleneck_dim + input_length, self.CNN_embed_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_p)
        )
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        inputs = {}
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        x = self.cnn(x)
        x = x.view(-1, 512*8*15)
        x = F.relu(self.bottleneck(x))
        if self.input_sets:
            for k, v in inputs.items():
                extra_input = getattr(self,'{}_input'.format(k))(v.to(torch.int64))
                x = torch.cat((x, extra_input), 1)

        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.classifier(x)
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = F.log_softmax(getattr(self,'{}_output'.format(k))(x), 1)
        return x_outs


class GameNormCNN(nn.Module):
    def __init__(self, sets, input_sets):
        super(GameNormCNN, self).__init__()
        cnn = nn.Sequential()
        self.sets = sets
        self.input_sets = input_sets
        self.drop_p = 0.3
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
        input_length = 0
        if self.input_sets:
            self.fc1 = nn.Linear(512*8*15, 256)
            self.bn1 = nn.BatchNorm1d(256)
            for k, v in self.input_sets.items():
                setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
                input_length += 100
            self.fc2 = nn.Linear(256 + input_length, self.CNN_embed_dim)
            self.bn2 = nn.BatchNorm1d(self.CNN_embed_dim)
        else:
            self.fc1 = nn.Linear(512*8*15, self.CNN_embed_dim)
            self.bn1 = nn.BatchNorm1d(self.CNN_embed_dim)
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        inputs = {}
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        x = self.cnn(x)
        x = x.view(-1, 512*8*15)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        if self.input_sets:
            for k, v in inputs.items():
                extra_input = getattr(self, '{}_input'.format(k))(v.to(torch.int64))
                x = torch.cat((x, extra_input), 1)
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = F.log_softmax(getattr(self,'{}_output'.format(k))(x), 1)
        return x_outs


class KillFeedCNN(nn.Module):
    def __init__(self, sets, input_sets):
        super(KillFeedCNN, self).__init__()
        self.sets = sets
        self.input_sets = input_sets
        self.drop_p = 0.3
        cnn = nn.Sequential()
        imgH = 32
        nc = 3
        self.CNN_embed_dim = 300
        self.bottleneck_dim = 256
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
        self.bottleneck = nn.Linear(512*75, self.bottleneck_dim)
        if self.input_sets:
            input_length = 0
            for k, v in self.input_sets.items():
                setattr(self, '{}_input'.format(k), nn.Embedding(len(v), 100))
                input_length += 100
            self.classifier = nn.Sequential(
                nn.Linear(self.bottleneck_dim + input_length, self.CNN_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.drop_p)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.bottleneck_dim, self.CNN_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.drop_p)
            )
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(self.CNN_embed_dim, len(v)))

    def forward(self, x):
        inputs = {}
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        x = self.cnn(x)
        x = x.view(-1, 512*75)
        x = F.relu(self.bottleneck(x))
        if self.input_sets:
            for k, v in inputs.items():
                extra_input = getattr(self,'{}_input'.format(k))(v.to(torch.int64))
                x = torch.cat((x, extra_input), 1)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.classifier(x)
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = F.log_softmax(getattr(self,'{}_output'.format(k))(x), dim=1)
        return x_outs
