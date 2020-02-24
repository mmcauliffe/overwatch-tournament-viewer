import torch.nn as nn
import torch
from annotator.datasets.ctc_dataset import LabelConverter
from annotator.game_values import HERO_SET, COLOR_SET, ABILITY_SET, KILL_FEED_INFO


ability_mapping = KILL_FEED_INFO['ability_mapping']
npc_set = KILL_FEED_INFO['npc_set']
deniable_ults = KILL_FEED_INFO['deniable_ults']
denying_abilities = KILL_FEED_INFO['denying_abilities']
npc_mapping = KILL_FEED_INFO['npc_mapping']


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
        self.classifier = nn.Sequential(
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
        output = self.classifier(conv)

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
        self.classifier = nn.Sequential(
            BidirectionalLSTM(512, self.num_hidden, self.num_hidden),
            BidirectionalLSTM(self.num_hidden, self.num_hidden, len(label_set) + 1))

    def forward(self, image):
        # conv features
        conv = self.cnn(image)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.classifier(conv)

        return output

class KillFeedCRNN(CRNN):
    num_hidden = 256

    def __init__(self, label_set):
        self.label_set = label_set
        self.converter = LabelConverter(label_set)
        super(KillFeedCRNN, self).__init__(32, 3, len(label_set) + 1, self.num_hidden)
        self.classifier = nn.Sequential(
            BidirectionalLSTM(512, self.num_hidden, self.num_hidden),
            BidirectionalLSTM(self.num_hidden, self.num_hidden, len(label_set) + 1))

    def forward(self, image):
        # conv features
        conv = self.cnn(image)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.classifier(conv)
        return output


class SideKillFeedCRNN(CRNN):
    num_hidden = 256

    def __init__(self, label_set):
        self.label_set = label_set
        self.converter = LabelConverter(label_set)
        super(SideKillFeedCRNN, self).__init__(32, 3, len(label_set) + 1, self.num_hidden)
        self.classifier = nn.Sequential(
            BidirectionalLSTM(512 + 6, self.num_hidden, self.num_hidden),
            BidirectionalLSTM(self.num_hidden, self.num_hidden, len(label_set) + 1))

    def forward(self, image, left_colors, right_colors):
        # conv features
        conv = self.cnn(image)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        left_colors = left_colors.unsqueeze(0)
        left_colors = left_colors.expand(conv.size(0), left_colors.size(1), left_colors.size(2))
        right_colors = right_colors.unsqueeze(0)
        right_colors = right_colors.expand(conv.size(0), right_colors.size(1), right_colors.size(2))
        combined = torch.cat((conv, left_colors), 2)
        combined = torch.cat((combined, right_colors), 2)
        # rnn features
        output = self.classifier(combined)

        return output

    def parse_image(self, image, left_colors, right_colors):
        batch_size = image.size(0)
        preds = self(image, left_colors, right_colors)
        pred_size = preds.size(0)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds = preds.to('cpu')

        return_data = []
        for s in range(batch_size):
            start_ind = s * pred_size
            end_ind = (s + 1) * pred_size
            d = [self.label_set[x - 1] for x in preds[start_ind:end_ind] if x != 0]
            data = {'first_hero': 'n/a',
                    'first_side': 'neither',
                    'assisting_heroes': [],
                    'ability': 'n/a',
                    'environmental': False,
                    'headshot': 'n/a',
                    'second_hero': 'n/a',
                    'second_side': 'n/a'}
            first_intervals = []
            second_intervals = []
            ability_intervals = []
            for i in d:
                if i == 'environmental':
                    data['environmental'] = True
                    continue
                if i in ABILITY_SET:
                    ability_intervals.append(i)
                if not len(ability_intervals):
                    first_intervals.append(i)
                elif i not in ABILITY_SET:
                    second_intervals.append(i)
            for i in first_intervals:
                if i in ['left', 'right']:
                    data['first_side'] = i
                elif i in HERO_SET:
                    data['first_hero'] = i
                else:
                    assist = i.replace('_assist', '')
                    if assist not in data['assisting_heroes'] and assist != data['first_hero'] and not i.endswith('npc'):
                        data['assisting_heroes'].append(assist)
            for i in ability_intervals:
                if i.endswith('headshot'):
                    data['headshot'] = True
                    data['ability'] = i.replace(' headshot', '')
                else:
                    data['ability'] = i
                    data['headshot'] = False
            for i in second_intervals:
                i = i.replace('_assist', '')
                if i in ['left', 'right']:
                    data['second_side'] = i
                elif i.endswith('_npc'):
                    data['second_hero'] = i.replace('_npc', '')
                elif i in HERO_SET:
                    data['second_hero'] = i
            if data['first_hero'] != 'n/a':
                if data['ability'] not in ability_mapping[data['first_hero']]:
                    data['ability'] = 'primary'
            return_data.append(data)
        return return_data