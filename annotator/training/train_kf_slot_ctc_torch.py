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
from torch.nn import CTCLoss
from annotator.models import crnn

working_dir = r'E:\Data\Overwatch\models\kill_feed_ctc'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')
TEST = True
train_dir = r'E:\Data\Overwatch\training_data\kill_feed_ctc_filtered'

# params

cuda = True
seed = 1
batch_size = 100
test_batch_size = 100
num_epochs = 10
lr = 0.001 # learning rate for Critic, not used by adadealta
beta1 = 0.5 # beta1 for adam. default=0.5
use_adam = False # whether to use adam (default is rmsprop)
use_adadelta = False # whether to use adadelta (default is rmsprop)
momentum = 0.5
log_interval = 10
image_height = 32
image_width = 248
num_channels = 3
num_hidden = 256
n_test_disp = 10
display_interval = 100
manualSeed = 1234 # reproduce experiemnt

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

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

#for i, lab in enumerate(labels):
#    if lab == '':
#        blank_ind = i
#        break
#class_count = blank_ind + 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + ['-']  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        '''
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
        '''
        length = []
        result = []
        for item in text:
            try:
                item = item.decode('utf-8', 'strict')
                if ' ' in item:
                    item = item.split(' ')
                length.append(len(item))
                for char in item:
                    index = self.dict[char]
                    result.append(index)
            except:
                print(item)
                raise
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ' '.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ' '.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

class TrainDataset(data.Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        self.data_num = 0
        self.data_indices = {}
        count = 0
        for f in os.listdir(train_dir):
            #if f != '8184.hdf5':
            #    continue
            if f.endswith('.hdf5'):
                with h5py.File(os.path.join(train_dir, f), 'r') as h5f:
                    self.data_num += h5f['train_img'].shape[0]
                    self.data_indices[self.data_num] = os.path.join(train_dir, f)
                count += 1
                if count > 2:
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

            labs[labs > blank_ind] = blank_ind
            labs = labs.reshape(labs.shape[0] * labs.shape[1])
            labs = labs[labs != blank_ind]
            labs += 1
            outputs['the_labels'] = torch.from_numpy(labs).long()
            outputs['label_length'] = torch.from_numpy(hf5["train_label_sequence_length"][real_index:real_index+batch_size]).long()
            # For removing all blank images
            #inds = outputs['label_length'] != 1
            #inputs['image'] = inputs['image'][inds]
            #outputs['the_labels'] = outputs['the_labels'][inds]
            #outputs['label_length'] = outputs['label_length'][inds]
            #for i in range(inputs['image'].shape[0]):
            #    length = outputs['label_length'][i]
            #    if blank_ind in outputs['the_labels'][i][:length]:
            #        print(outputs['the_labels'][i])
            #        print(outputs['the_labels'][i][:length])
            #        print(outputs['label_length'][i])
            #        raise Exception

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
                if count > 2:
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
            labs = hf5["val_label_sequence"][real_index:real_index+batch_size, ...].astype(np.int32)
            labs[labs > blank_ind] = blank_ind
            labs = labs.reshape(labs.shape[0] * labs.shape[1])
            labs = labs[labs != blank_ind]
            labs += 1
            outputs['the_labels'] = torch.from_numpy(labs).long()
            outputs['label_length'] = torch.from_numpy(np.reshape(hf5["val_label_sequence_length"][real_index:real_index+batch_size], (-1, 1))).long()
        return inputs, outputs

    def __len__(self):
        return int(self.data_num / test_batch_size)

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

train_set = TrainDataset()
test_set = TestDataset()


# net init
# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

num_classes = len(labels) + 1
net = crnn.CRNN(image_height, num_channels, num_classes, num_hidden)
net.apply(weights_init)
print(net)

# -------------------------------------------------------------------------------------------------
converter = strLabelConverter(labels)
criterion = CTCLoss()

image = torch.FloatTensor(batch_size, 3, image_height, image_width)
text = torch.IntTensor(batch_size * 5)
length = torch.IntTensor(batch_size)
if cuda and torch.cuda.is_available():
    net.cuda()
    image = image.cuda()
    criterion = criterion.cuda()
image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = averager()

# setup optimizer
if use_adam:
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
elif use_adadelta:
    optimizer = optim.Adadelta(net.parameters())
else:
    optimizer = optim.RMSprop(net.parameters(), lr=lr)


def loadData(v, data):
    v.resize_(data.size()).copy_(data)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=True)

    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        inputs, outputs = val_iter.next()
        i += 1
        cpu_images = inputs['image'][0]
        cpu_texts = outputs['the_labels'][0]
        cpu_lengths = outputs['label_length'][0]

        batch_size = cpu_images.size(0)
        loadData(image, cpu_images)
        loadData(text, cpu_texts)
        loadData(length, cpu_lengths)

        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = converter.decode(cpu_texts, cpu_lengths, raw=False)
        for pred, target in zip(sim_preds, cpu_texts_decode):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * batch_size)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):
    inputs, outputs = train_iter.next()
    cpu_images = inputs['image'][0]
    cpu_texts = outputs['the_labels'][0]
    cpu_lengths = outputs['label_length'][0]
    batch_size = cpu_images.size(0)
    loadData(image, cpu_images)

    loadData(text, cpu_texts)
    loadData(length, cpu_lengths)

    optimizer.zero_grad()
    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

    cost = criterion(preds, text, preds_size, length) / batch_size
    # net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


### TRAINING

if __name__ == '__main__':
    import time
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                              shuffle=True)
    print(len(train_loader), 'batches')
    for epoch in range(num_epochs):
        print('Epoch', epoch)
        begin = time.time()
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            for p in net.parameters():
                p.requires_grad = True
            net.train()

            cost = trainBatch(net, criterion, optimizer)
            loss_avg.add(cost)
            i += 1

            if i % display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        val(net, test_set, criterion)

        # do checkpointing
        torch.save(net.state_dict(), os.path.join(working_dir, 'netCRNN_{}.pth'.format(epoch)))
        print('Time per epoch: {} seconds'.format(time.time()-begin))