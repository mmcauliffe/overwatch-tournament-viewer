import os
import torch
import numpy as np
import torch.optim as optim
import random
from annotator.datasets.lstm_cnn_dataset import LstmCnnHDF5Dataset
import torch.nn as nn
from annotator.models.rnn import StatusGRU
from annotator.training.lstm_helper import train_batch, val
from annotator.training.helper import Averager, load_set

TEST = True

working_dir = r'E:\Data\Overwatch\models\player_rnn'
train_dir = r'E:\Data\Overwatch\training_data\player_lstm'
cnn_model = r'E:\Data\Overwatch\models\player_status\model.pth'
lstm_model = os.path.join(working_dir, 'model.pth')

cuda = True
seed = 1
batch_size = 4
test_batch_size = 4
num_epochs = 10
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
display_interval = 10
manualSeed = 1234 # reproduce experiemnt
random_sample = True
recent = False # Only use recent rounds

load_CNN_pretrained = True
train_CNN = True

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')

if __name__ == '__main__':
    input_set_files = {
        'color': os.path.join(train_dir, 'color_set.txt'),
        'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
         }
    input_sets = {}
    for k, v in input_set_files.items():
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
        sets[k] = load_set(v)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set = LstmCnnHDF5Dataset(train_dir, sets=sets, input_sets=input_sets, batch_size=batch_size, pre='train', recent=recent)
    test_set = LstmCnnHDF5Dataset(train_dir, sets=sets, input_sets=input_sets, batch_size=test_batch_size, pre='val', recent=recent)
    weights = train_set.generate_class_weights(mu=10)

    net = StatusGRU(sets, input_sets)
    crnn_params = net.parameters()
    net.to(device)

    start_epoch = 0
    if load_CNN_pretrained:
        if False and os.path.exists(lstm_model): # Initialize from CNN model
            d = torch.load(lstm_model)
            print('LOADED RNN MODEL')
            net.load_state_dict(d, strict=False)
            start_epoch = 1
        elif os.path.exists(cnn_model): # Initialize from CNN model
            d = torch.load(cnn_model)
            net.cnn.load_state_dict(d, strict=False)
            print('LOADED CNN MODEL')

    print('WEIGHTS')
    for k, v in weights.items():
        print(k)
        print(', '.join('{}: {}'.format(sets[k][k2], v2) for k2, v2 in enumerate(v)))

    losses = {}
    for k in sets.keys():
        losses[k] = nn.CrossEntropyLoss(weight=weights[k])
        losses[k].to(device)

    if use_adam:
        optimizer = optim.Adam(crnn_params, lr=lr, betas=(beta1, 0.999))
    elif use_adadelta:
        optimizer = optim.Adadelta(crnn_params)
    else:
        optimizer = optim.RMSprop(crnn_params, lr=lr)

    # loss averager
    loss_avg = Averager()
    import time
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=num_workers)

    print(len(train_loader), 'batches')
    best_val_loss = np.inf
    for epoch in range(start_epoch, num_epochs):
        print('Epoch', epoch)
        begin = time.time()
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            net.train()
            for p in net.parameters():
                p.requires_grad = True
            if epoch <= 2:
                net.cnn.eval()
                for p in net.cnn.parameters():
                    p.requires_grad = False
            else:
                net.cnn.train()
                for p in net.cnn.parameters():
                    p.requires_grad = True

            cost = train_batch(net, train_iter, device, losses, optimizer)
            loss_avg.add(cost)
            i += 1
            if i % display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        best_val_loss = val(net, val_loader, device, losses, working_dir, best_val_loss)

        # do checkpointing
        #torch.save(cnn_encoder.state_dict(), os.path.join(working_dir, 'netCNN_{}.pth'.format(epoch)))
        #torch.save(rnn_decoder.state_dict(), os.path.join(working_dir, 'netRNN_{}.pth'.format(epoch)))
        torch.save(net.state_dict(), os.path.join(working_dir, 'netRNN_{}.pth'.format(epoch)))
        print('Time per epoch: {} seconds'.format(time.time() - begin))
        train_set.file_index = 0
        train_set.sequence_index = 0
        train_set.slot_index = 0
        test_set.file_index = 0
        test_set.sequence_index = 0
        test_set.slot_index = 0
    print('Completed training, best val loss was: {}'.format(best_val_loss))


