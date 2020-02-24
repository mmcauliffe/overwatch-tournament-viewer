import os
import shutil
import torch
import argparse
import numpy as np
from torch import nn

import torch.optim as optim
import random
from dataset import CNNHDF5Dataset, test_dataset
from utils import Averager, load_set, train_batch, val
from models import StatusCNN

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
parser.add_argument('--num_epochs', type=int, default=25,
                    help='number of epochs to train')
parser.add_argument('--num_files', type=int, default=100,
                    help='number of files to load')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='beta')
parser.add_argument('--early_stopping', type=int, default=2, help='Number of iterations to stop without an improvement')
parser.add_argument('--debug', type=bool, default=False, help='Debug dataset before running')
parser.add_argument('--recent', type=bool, default=False, help='Only recent files')
args = parser.parse_args()

train_dir = args.data_folder
working_dir = args.output_dir
train_dir = os.path.join(train_dir, os.listdir(train_dir)[0])
print('Data folder:', train_dir)
print(os.listdir(train_dir))
print(os.listdir(os.path.dirname(train_dir)))
if args.debug:
    test_dataset(train_dir)

os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')
TEST = True

# params

cuda = True
seed = 1
batch_size = 100
test_batch_size = 100
num_epochs = 20
lr = args.learning_rate # learning rate for Critic, not used by adadealta
beta1 = args.beta # beta1 for adam. default=0.5
use_adam = True # whether to use adam (default is rmsprop)
use_adadelta = False # whether to use adadelta (default is rmsprop)
early_stopping_threshold = args.early_stopping
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
num_files = args.num_files
recent = args.recent

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

model_path = os.path.join(working_dir, 'model.pth')

### TRAINING

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

if __name__ == '__main__':
    input_set_files = {
        'color': os.path.join(train_dir, 'color_set.txt'),
        'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
         }
    input_sets = {}
    for k, v in input_set_files.items():
        shutil.copyfile(v, os.path.join(working_dir, '{}_set.txt'.format(k)))
        input_sets[k] = load_set(v)

    set_files = {  # 'player': os.path.join(train_dir, 'player_set.txt'),
        'hero': os.path.join(train_dir, 'hero_set.txt'),
        'alive': os.path.join(train_dir, 'alive_set.txt'),
        'ult': os.path.join(train_dir, 'ult_set.txt'),
        'switch': os.path.join(train_dir, 'switch_set.txt'),
        'status': os.path.join(train_dir, 'status_set.txt'),
        'antiheal': os.path.join(train_dir, 'antiheal_set.txt'),
        'immortal': os.path.join(train_dir, 'immortal_set.txt'),
    }
    sets = {}
    for k, v in set_files.items():
        shutil.copyfile(v, os.path.join(working_dir, '{}_set.txt'.format(k)))
        sets[k] = load_set(v)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set = CNNHDF5Dataset(train_dir, sets=sets, input_sets=input_sets, batch_size=batch_size, pre='train', num_files=num_files, recent=recent)
    test_set = CNNHDF5Dataset(train_dir, sets=sets, input_sets=input_sets, batch_size=test_batch_size, pre='val', num_files=num_files, recent=recent)
    weights = train_set.generate_class_weights(mu=10)
    print(len(train_set))

    net = StatusCNN(sets, input_sets)
    net.to(device)

    if os.path.exists(model_path): # Initialize from CNN model
        d = torch.load(model_path)
        net.load_state_dict(d, strict=False)
        print('Loaded previous model')

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

    check_point_path = os.path.join(working_dir, 'checkpoint.pth')
    net, optimizer, start_epoch, best_val_loss = load_checkpoint(net, optimizer, check_point_path)
    last_improvement = start_epoch
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
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'best_val_loss': best_val_loss}
        torch.save(state, check_point_path)
        print('Time per epoch: {} seconds'.format(time.time() - begin))
        if epoch - last_improvement == early_stopping_threshold:
            print('Stopping training, val loss hasn\'t improved in {} iterations.'.format(early_stopping_threshold))
            break
    print('Completed training, best val loss was: {}'.format(best_val_loss))


