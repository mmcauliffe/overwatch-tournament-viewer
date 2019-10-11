import os
import shutil
import torch
import numpy as np

import torch.optim as optim
import random
from annotator.datasets.cnn_dataset import CNNDataset, BatchedCNNDataset, CNNHDF5Dataset
from annotator.datasets.helper import randomSequentialSampler
import torch.nn as nn
from annotator.models.cnn import StatusCNN
from annotator.training.cnn_helper import train_batch, val
from annotator.training.helper import Averager, load_set

TEST = True
train_dir = r'E:\Data\Overwatch\training_data\player_status'

cuda = True
seed = 1
batch_size = 300
test_batch_size = 500
num_epochs = 5
lr = 0.001 # learning rate for Critic, not used by adadealta
momentum = 0.5
beta1 = 0.5 # beta1 for adam. default=0.5
use_adam = True # whether to use adam (default is rmsprop)
use_adadelta = False # whether to use adadelta (default is rmsprop)
log_interval = 10
image_height = 64
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
recent = False # Only use recent rounds

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

working_dir = r'E:\Data\Overwatch\models\player_status'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')
model_path = os.path.join(working_dir, 'model.pth')

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

    if use_hdf5:
        train_set = CNNHDF5Dataset(train_dir, sets=sets, input_sets=input_sets, batch_size=batch_size, pre='train', recent=recent)
        test_set = CNNHDF5Dataset(train_dir, sets=sets, input_sets=input_sets, batch_size=test_batch_size, pre='val', recent=recent)
        weights = train_set.generate_class_weights(mu=10)
        print(len(train_set))
    else:
        if use_batched_dataset:
                train_set = BatchedCNNDataset(root=os.path.join(train_dir, 'training_set'), sets=sets, batch_size=batch_size)
                test_set = BatchedCNNDataset(root=os.path.join(train_dir, 'val_set'), sets=sets, batch_size=test_batch_size)
        else:
            train_set = CNNDataset(root=os.path.join(train_dir, 'training_set'), sets=sets)
            if not random_sample:
                sampler = randomSequentialSampler(train_set, batch_size)
            else:
                sampler = None
            test_set = CNNDataset(root=os.path.join(train_dir, 'val_set'), sets=sets)
        weights = train_set.generate_class_weights(mu=10, train_directory=train_dir)

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

            cost = train_batch(net, train_iter, device, losses, optimizer, use_batched_dataset=use_batched_dataset)
            loss_avg.add(cost)
            i += 1
            if i % display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        best_val_loss = val(net, val_loader, device, losses, working_dir, best_val_loss, use_batched_dataset=use_batched_dataset)

        # do checkpointing
        torch.save(net.state_dict(), os.path.join(working_dir, 'netCNN_{}.pth'.format(epoch)))
        print('Time per epoch: {} seconds'.format(time.time() - begin))
    print('Completed training, best val loss was: {}'.format(best_val_loss))


