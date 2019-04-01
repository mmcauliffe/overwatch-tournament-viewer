import os
import torch
#import torchvision
#import torchvision.transforms as transforms
import h5py
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import random
import itertools

working_dir = r'E:\Data\Overwatch\models\kill_feed_ctc'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')
TEST = True
train_dir = r'E:\Data\Overwatch\training_data\kill_feed_ctc'

cuda = True
seed = 1
batch_size = 100
test_batch_size = 100
epochs = 10
lr = 0.01
momentum = 0.5
log_interval = 10


def labels_to_text(ls):
    ret = []
    for c in ls:
        if c >= len(labels):
            continue
        ret.append(labels[c])
    return ret


def decode_batch(out):
    ret = []
    print(out.shape)
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        print(out_best)
        out_best = [k for k, g in itertools.groupby(out_best)]
        print(out_best)
        outstr = labels_to_text(out_best)
        print(outstr)
        ret.append(outstr)
    return ret


def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


labels = load_set(os.path.join(train_dir, 'labels_set.txt'))

class_count = len(labels)
blank_ind = len(labels) - 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class TrainDataset(data.Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        self.data_num = 0
        self.data_indices = {}
        count = 0
        for f in os.listdir(train_dir):
            if f != '8184.hdf5':
                continue
            if f.endswith('.hdf5'):
                with h5py.File(os.path.join(train_dir, f), 'r') as h5f:
                    self.data_num += h5f['train_img'].shape[0]
                    self.data_indices[self.data_num] = os.path.join(train_dir, f)
                count += 1
                if count > 1:
                    break
        self.weights = {}
        print('DONE SETTING UP')

    def __len__(self):
        return int(self.data_num / batch_size)

    def __getitem__(self, index):
        start_ind = 0
        real_index = index * batch_size
        for i, (next_ind, v) in enumerate(self.data_indices.items()):
            path = v
            if real_index < next_ind:
                break
            start_ind = next_ind

        real_index = real_index - start_ind
        inputs = {}
        outputs = {}
        with h5py.File(path, 'r') as hf5:

            inputs['image']= torch.from_numpy(np.transpose(hf5['train_img'][real_index:real_index+batch_size, ...],  (0, 3 ,2, 1))).float()
            labs = hf5["train_label_sequence"][real_index:real_index+batch_size, ...].astype(np.int32)
            #labs += 1
            labs[labs > blank_ind] = blank_ind
            outputs['the_labels'] = torch.from_numpy(labs).long()
            print(hf5["train_label_sequence_length"][real_index:real_index+batch_size].shape)
            outputs['label_length'] = torch.from_numpy(hf5["train_label_sequence_length"][real_index:real_index+batch_size]).long()
            print(outputs['label_length'].shape)
            # For removing all blank images
            inds = outputs['label_length'] != 1
            inputs['image'] = inputs['image'][inds]
            outputs['the_labels'] = outputs['the_labels'][inds]
            outputs['label_length'] = outputs['label_length'][inds]
            for i in range(inputs['image'].shape[0]):
                length = outputs['label_length'][i]
                if length > 1 and blank_ind in outputs['the_labels'][i][:length]:
                    print(outputs['the_labels'][i])
                    print(outputs['the_labels'][i][:length])
                    print(outputs['label_length'][i])
                    error

        return inputs, outputs


class TestDataset(data.Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        self.data_num = 0
        self.data_indices = {}
        count=0
        for f in os.listdir(train_dir):
            if f.endswith('.hdf5'):
                with h5py.File(os.path.join(train_dir, f), 'r') as h5f:
                    self.data_num += h5f['val_img'].shape[0]
                    self.data_indices[self.data_num] = os.path.join(train_dir, f)
                count += 1
                if count > 1:
                    break

    def __getitem__(self, index):
        start_ind = 0
        real_index = index * test_batch_size
        for i, (next_ind, v) in enumerate(self.data_indices.items()):
            path = v
            if real_index < next_ind:
                break
            start_ind = next_ind
        real_index = real_index - start_ind
        inputs = {}
        outputs = {}
        with h5py.File(path, 'r') as hf5:
            inputs['image']= torch.from_numpy(np.transpose(hf5['val_img'][real_index:real_index+batch_size, ...], (0, 3 ,2, 1))).float()

            outputs['the_labels'] = torch.from_numpy(hf5["val_label_sequence"][real_index:real_index+batch_size, ...].astype(np.int32)).long()
            outputs['label_length'] = torch.from_numpy(np.reshape(hf5["val_label_sequence_length"][real_index:real_index+batch_size], (-1, 1))).long()
        return inputs, outputs

    def __len__(self):
        return int(self.data_num / test_batch_size)


train_set = TrainDataset()
trainloader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                          shuffle=True)
test_set = TestDataset()
testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=True)

def imshow(img):
    img = img.cpu()
    npimg = img.numpy()
    img = np.transpose(npimg, (1,2, 0))[:,:, [2,1,0]]/255
    plt.imshow(img)
    plt.show()

# get some random training images
dataiter = iter(trainloader)
inputs, outputs = dataiter.next()
print(inputs['image'].shape)
print(outputs['the_labels'].shape)
# show images
#imshow(torchvision.utils.make_grid(inputs['image'][0, :4, ...],nrow=1))

for i in range(4):
    print(outputs['the_labels'][0,i])
    print('Labels', ' '.join(labels[x] for x in outputs['the_labels'][0,i] if x < len(labels)))


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
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

    def forward(self, input):
        input = input['image']
        # conv features
        conv = self.cnn(input)
        print(conv.shape)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        print(conv.shape)

        # rnn features
        output = self.rnn(conv)
        print(output.shape)
        return output


net = CRNN(32, 3, len(labels), nh=256)
net.to(device)

ctc_loss = nn.CTCLoss(blank=blank_ind)
ctc_loss.to(device)
#optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
optimizer = optim.Adagrad(net.parameters())

import time
for epoch in range(2):  # loop over the dataset multiple times
    batch_begin = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        print(i)
        # get the inputs
        #begin = time.time()
        inputs, data_labels = data
        for k,v in inputs.items():
            v = v[0]
            inputs[k] = v.to(device)
        for k,v in data_labels.items():
            v = v[0]
            data_labels[k] = v.to(device)
        #print('Loading data took: {}'.format(time.time()-begin))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #begin = time.time()
        predicteds = net(inputs)
        print(predicteds.shape)
        predicteds = predicteds.log_softmax(2)
        print(predicteds.shape)
        input_lengths = torch.full((inputs['image'].shape[0],),
                                   predicteds.shape[0]).long()
        input_lengths.to(device)

        #print('Predicting data took: {}'.format(time.time()-begin))
        decode_batch(predicteds.cpu().detach().numpy())
        #begin = time.time()
        print(predicteds.shape)
        #print('TRUE LABELS', data_labels['the_labels'].shape)
        #print('TRUE LABELS', data_labels['the_labels'])
        #print(input_lengths)
        #print(input_lengths.shape)
        #print(data_labels['label_length'])
        #print(data_labels['label_length'].shape)
        loss = ctc_loss(predicteds, data_labels['the_labels'], input_lengths, data_labels['label_length'])
        print(loss.item())

        #print('Loss calculation took: {}'.format(time.time()-begin))
        #begin = time.time()
        loss.backward()
        optimizer.step()
        #print('Back prop took: {}'.format(time.time()-begin))

        # print statistics
        #begin = time.time()
        running_loss += loss.item()
        #print(loss.item())
        if i % 50 == 49:    # print every 50 mini-batches
            #print(predicteds['hero'])
            #print(labels['hero'])
            print('Epoch %d, %d/%d, loss: %.3f' %
                  (epoch + 1, i + 1, len(train_set), running_loss / i))
            running_loss = 0.0
            print('Batch took: {}'.format(time.time()-batch_begin))
        batch_begin = time.time()



print('Finished Training')

dataiter = iter(testloader)
inputs, outputs = dataiter.next()
for k,v in inputs.items():
    v = v[0]
    inputs[k] = v.to(device)
for k,v in outputs.items():
    v = v[0]
    outputs[k] = v.to(device)

# print images
#imshow(torchvision.utils.make_grid(inputs['image'][:4, ...], nrow=1))
with torch.no_grad():
    predicteds = net(inputs)
    predicted_labels = decode_batch(predicteds.cpu())
    for i in range(4):
        print(predicted_labels[i])
        print('Labels', ' '.join(labels[x] for x in outputs['the_labels'][i] if x < len(labels)))
