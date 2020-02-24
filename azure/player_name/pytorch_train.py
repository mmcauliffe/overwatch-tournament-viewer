import os
import torch
import shutil
#import torchvision
#import torchvision.transforms as transforms
import numpy as np
import argparse
import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import random
from torch.nn import CTCLoss
from dataset import CTCHDF5Dataset, LabelConverter
from utils import Averager, load_set, train_batch, val
from models import KillFeedCRNN

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--regularization', type=float, dest='reg', default=0.01, help='regularization rate')
parser.add_argument('--num_epochs', type=int, default=25,
                    help='number of epochs to train')
parser.add_argument('--output_dir', type=str, help='output directory')
parser.add_argument('--learning_rate', type=float,
                    default=0.001, help='learning rate')
parser.add_argument('--beta', type=float, default=0.5, help='beta')
parser.add_argument('--early_stopping', type=int, default=2, help='Number of iterations to stop without an improvement')
args = parser.parse_args()

train_dir = args.data_folder
working_dir = args.output_dir
#train_dir = os.path.join(train_dir, os.listdir(train_dir)[0])
print('Data folder:', train_dir)
print(os.listdir(train_dir))
print(os.listdir(os.path.dirname(train_dir)))

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
    net = KillFeedCRNN(label_set, spectator_mode_set)
    net.apply(weights_init)
    print(net)

    # -------------------------------------------------------------------------------------------------
    converter = LabelConverter(label_set)
    criterion = CTCLoss(zero_infinity=True)

    image = torch.FloatTensor(batch_size, 3, image_height, image_width)
    spectator_modes = torch.IntTensor(batch_size)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)
    if cuda and torch.cuda.is_available():
        net.cuda()
        image = image.cuda()
        spectator_modes = spectator_modes.cuda()
        criterion = criterion.cuda()
        #text = text.cuda()
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

            cost = train_batch(net, train_iter, device, criterion, optimizer,image, spectator_modes, text, length)
            i += 1
            if cost is None:
                continue
            loss_avg.add(cost)

            if i % display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        prev_best = best_val_loss
        best_val_loss = val(net, val_loader, device, criterion, working_dir, best_val_loss, converter,image, spectator_modes, text, length)
        if best_val_loss < prev_best:
            last_improvement = epoch

        # do checkpointing
        torch.save(net.state_dict(), os.path.join(working_dir, 'netCRNN_{}.pth'.format(epoch)))
        print('Time per epoch: {} seconds'.format(time.time()-begin))
        if epoch - last_improvement == early_stopping_threshold:
            print('Stopping training, val loss hasn\'t improved in {} iterations.'.format(early_stopping_threshold))
            break
    print('Completed training, best val loss was: {}'.format(best_val_loss))