import os
import torch
import torchvision
import torchvision.transforms as transforms
import h5py
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
import random
from annotator.datasets.cnn_dataset import CNNHDF5Dataset
from annotator.datasets.helper import randomSequentialSampler
import torch.nn as nn
from annotator.training.cnn_helper import train_batch, val
from annotator.training.helper import Averager, load_set
from annotator.models.cnn import MidCNN

working_dir = r'E:\Data\Overwatch\models\mid'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')
TEST = True
train_dir = r'E:\Data\Overwatch\training_data\mid'

cuda = True
seed = 1
batch_size = 200
test_batch_size = 400
num_epochs = 20
early_stopping_threshold = 2
lr = 0.001 # learning rate for Critic, not used by adadealta
momentum = 0.5
beta1 = 0.5 # beta1 for adam. default=0.5
use_adam = True # whether to use adam (default is rmsprop)
use_adadelta = False # whether to use adadelta (default is rmsprop)
log_interval = 10
image_height = 48
image_width = 144
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

model_path = os.path.join(working_dir, 'model.pth')


if __name__ == '__main__':
    set_files = {
        'overtime': os.path.join(train_dir, 'overtime_set.txt'),
        'point_status': os.path.join(train_dir, 'point_status_set.txt'),
        'attacking_side': os.path.join(train_dir, 'attacking_side_set.txt'),
        'map': os.path.join(train_dir, 'map_set.txt'),
        'map_mode': os.path.join(train_dir, 'map_mode_set.txt'),
        'round_number': os.path.join(train_dir, 'round_number_set.txt'),
        'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
    }

    sets = {}
    for k, v in set_files.items():
        sets[k] = load_set(v)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set = CNNHDF5Dataset(train_dir, sets=sets, batch_size=batch_size, pre='train')
    test_set = CNNHDF5Dataset(train_dir, sets=sets, batch_size=test_batch_size, pre='val')
    weights = train_set.generate_class_weights(mu=10)
    print(len(train_set))

    net = MidCNN(sets)
    if os.path.exists(model_path): # Initialize from CNN model
        d = torch.load(model_path)
        net.load_state_dict(d, strict=False)
        print('Loaded previous model')
    net.to(device)

    print('WEIGHTS')
    for k, v in weights.items():
        print(k)
        print(', '.join('{}: {}'.format(sets[k][k2], v2) for k2, v2 in enumerate(v)))

    losses = {}
    for k in sets.keys():
        losses[k] = nn.CrossEntropyLoss(weight=weights[k])
        losses[k].to(device)

    if use_adam:
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    elif use_adadelta:
        optimizer = optim.Adadelta(net.parameters())
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=lr)

    # loss averager
    loss_avg = Averager()
    import time
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=num_workers,
                                              shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=True, num_workers=num_workers)
    print(len(train_loader), 'batches')
    best_val_loss = np.inf
    last_improvement = 0
    for epoch in range(num_epochs):
        print('Epoch', epoch)
        begin = time.time()
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            for p in net.parameters():
                p.requires_grad = True
            net.train()

            cost = train_batch(net, train_iter, device, losses, optimizer)
            loss_avg.add(cost)
            i += 1
            if i % display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        prev_best = best_val_loss
        best_val_loss = val(net, val_loader, device, losses, working_dir, best_val_loss)
        if best_val_loss < prev_best:
            last_improvement = epoch

        # do checkpointing
        torch.save(net.state_dict(), os.path.join(working_dir, 'netCNN_{}.pth'.format(epoch)))
        print('Time per epoch: {} seconds'.format(time.time() - begin))
        if epoch - last_improvement == early_stopping_threshold:
            print('Stopping training, val loss hasn\'t improved in {} iterations.'.format(early_stopping_threshold))
            break
    print('Completed training, best val loss was: {}'.format(best_val_loss))

