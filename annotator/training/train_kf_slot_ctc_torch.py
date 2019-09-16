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
from annotator.datasets.ctc_dataset import CTCDataset, CTCHDF5Dataset, LabelConverter
from annotator.datasets.helper import randomSequentialSampler
from annotator.training.helper import Averager, load_set
from annotator.models import crnn
from annotator.training.ctc_helper import train_batch, val

working_dir = r'E:\Data\Overwatch\models\kill_feed_ctc'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')

TEST = True
train_dir = r'E:\Data\Overwatch\training_data\kill_feed_ctc'

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
image_width = 248
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


def load_checkpoint(model, optimizer, filename='checkpoint'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val_loss = np.inf
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', np.inf)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, best_val_loss

### TRAINING

model_path = os.path.join(working_dir, 'model.pth')
if __name__ == '__main__':

    spectator_mode_set = load_set(os.path.join(train_dir, 'spectator_mode_set.txt'))
    label_set = load_set(os.path.join(train_dir, 'labels_set.txt'))
    for i, lab in enumerate(label_set):
        if not lab:
            blank_ind = i
            break
    else:
        blank_ind = len(label_set)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if use_hdf5:
        train_set = CTCHDF5Dataset(train_dir, batch_size, blank_ind, pre='train')
        test_set = CTCHDF5Dataset(train_dir, batch_size, blank_ind, pre='val')
        #weights = train_set.generate_class_weights(mu=10)
        print(len(train_set))
    else:
        train_set = CTCDataset(root=os.path.join(train_dir, 'training_set'))
        if not random_sample:
            sampler = randomSequentialSampler(train_set, batch_size)
        else:
            sampler = None
        test_set = CTCDataset(root=os.path.join(train_dir, 'val_set'))


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
    check_point_path = os.path.join(working_dir, 'checkpoint.pth')
    net, optimizer, start_epoch, best_val_loss = load_checkpoint(net, optimizer, check_point_path)

    if os.path.exists(model_path):  # Initialize from CNN model
        d = torch.load(model_path)
        print(d)
        net.load_state_dict(d, strict=False)
    net = net.to(device)
    # now individually transfer the optimizer parts...
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    for epoch in range(start_epoch, num_epochs):
        epoch_state_path = os.path.join(working_dir, 'netCRNN_{}.pth'.format(epoch))

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

        best_val_loss = val(net, val_loader, device, criterion, working_dir, best_val_loss, converter,image, spectator_modes,  text, length,
                                    use_batched_dataset=use_batched_dataset)

        # do checkpointing
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'best_val_loss': best_val_loss}
        torch.save(state, check_point_path)
        print('Time per epoch: {} seconds'.format(time.time()-begin))
    print('Completed training, best val loss was: {}'.format(best_val_loss))