import os
import torch
import shutil
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
from annotator.datasets.ctc_dataset import CTCHDF5Dataset, LabelConverter
from annotator.datasets.helper import randomSequentialSampler
from annotator.training.helper import Averager, load_set
from annotator.models import crnn
from annotator.training.ctc_helper import train_batch, val

working_dir = r'E:\Data\Overwatch\models\player_ocr'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')
TEST = True
train_dir = r'E:\Data\Overwatch\training_data\player_ocr'

# params

cuda = True
seed = 1
batch_size = 100
test_batch_size = 100
num_epochs = 10
lr = 0.001 # learning rate for Critic, not used by adadealta
beta1 = 0.5 # beta1 for adam. default=0.5
use_adam = True # whether to use adam (default is rmsprop)
use_adadelta = False # whether to use adadelta (default is rmsprop)
momentum = 0.5
log_interval = 10
image_height = 32
image_width = 64
num_channels = 3
num_hidden = 256
num_workers = 0
n_test_disp = 10
display_interval = 100
manualSeed = 1234 # reproduce experiemnt
random_sample = True
use_batched_dataset = True
use_hdf5 = True

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)



### TRAINING

if __name__ == '__main__':
    label_path = os.path.join(train_dir, 'labels_set.txt')
    label_set = load_set(label_path)
    spec_mode_path = os.path.join(train_dir, 'spectator_mode_set.txt')
    spectator_mode_set = load_set(spec_mode_path)
    shutil.copyfile(label_path, os.path.join(working_dir, 'labels_set.txt'))
    shutil.copyfile(spec_mode_path, os.path.join(working_dir, 'spectator_mode_set.txt'))
    for i, lab in enumerate(label_set):
        if not lab:
            blank_ind = i
            break
    else:
        blank_ind = len(label_set)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set = CTCHDF5Dataset(train_dir, batch_size, blank_ind, pre='train')
    test_set = CTCHDF5Dataset(train_dir, batch_size, blank_ind, pre='val')
    #weights = train_set.generate_class_weights(mu=10)
    print(len(train_set))


    # net init
    # custom weights initialization called on crnn
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    num_classes = len(label_set) + 1
    net = crnn.KillFeedCRNN(label_set, spectator_mode_set)
    net.apply(weights_init)
    print(net)

    # -------------------------------------------------------------------------------------------------
    converter = LabelConverter(label_set)
    criterion = CTCLoss()

    image = torch.FloatTensor(batch_size, 3, image_height, image_width)
    spectator_modes = torch.IntTensor(batch_size)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)
    if cuda and torch.cuda.is_available():
        net.cuda()
        image = image.cuda()
        spectator_modes = spectator_modes.cuda()
        criterion = criterion.cuda()
    image = Variable(image)
    spectator_modes = Variable(spectator_modes)
    text = Variable(text)
    length = Variable(length)

    # loss averager
    loss_avg = Averager()

    # setup optimizer
    if use_adam:
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    elif use_adadelta:
        optimizer = optim.Adadelta(net.parameters())
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=lr)
    import time
    if use_batched_dataset:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=num_workers,
                                                  shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                  shuffle=True, num_workers=num_workers)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                                                  shuffle=True)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size,
                                                  shuffle=True, num_workers=num_workers)
    print(len(train_loader), 'batches')
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        print('Epoch', epoch)
        begin = time.time()
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            for p in net.parameters():
                p.requires_grad = True
            net.train()

            cost = train_batch(net, train_iter, device, criterion, optimizer,image, spectator_modes, text, length, use_batched_dataset=use_batched_dataset)
            i += 1
            if cost is None:
                continue
            loss_avg.add(cost)

            if i % display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        best_val_loss = val(net, val_loader, device, criterion, working_dir, best_val_loss, converter,image, spectator_modes, text, length,
                                    use_batched_dataset=use_batched_dataset)

        # do checkpointing
        torch.save(net.state_dict(), os.path.join(working_dir, 'netCRNN_{}.pth'.format(epoch)))
        print('Time per epoch: {} seconds'.format(time.time()-begin))
    print('Completed training, best val loss was: {}'.format(best_val_loss))