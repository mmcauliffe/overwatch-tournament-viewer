import torch.nn as nn
import torch
from annotator.datasets.ctc_dataset import LabelConverter


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
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
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

        # replace all nan/inf in gradients to zero
        self.register_backward_hook(self.backward_hook)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0  # replace all nan/inf in gradients to zero


class PlayerNameCRNN(CRNN):
    num_hidden = 256

    def __init__(self, label_set):
        self.label_set = label_set
        self.converter = LabelConverter(label_set)
        super(PlayerNameCRNN, self).__init__(32, 3, len(label_set) + 1, self.num_hidden)


class KillFeedCRNN(CRNN):
    num_hidden = 256

    def __init__(self, label_set, spectator_mode_set):
        self.label_set = label_set
        self.converter = LabelConverter(label_set)
        self.spectator_mode_set = spectator_mode_set
        super(KillFeedCRNN, self).__init__(32, 3, len(label_set) + 1, self.num_hidden)
        self.spectator_mode_embedding = nn.Embedding(len(spectator_mode_set), len(spectator_mode_set))
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512 + len(spectator_mode_set), self.num_hidden, self.num_hidden),
            BidirectionalLSTM(self.num_hidden, self.num_hidden, len(label_set) + 1))

    def forward(self, image, spectator_mode):
        # conv features
        conv = self.cnn(image)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        spec = self.spectator_mode_embedding(spectator_mode.to(torch.int64))
        spec.unsqueeze_(0)
        spec = spec.expand(conv.size(0), spec.size(1), spec.size(2))

        combined = torch.cat((conv, spec), 2)
        # rnn features
        output = self.rnn(combined)

        return output
