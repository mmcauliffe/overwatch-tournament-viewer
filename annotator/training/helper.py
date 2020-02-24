import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os

lr = 0.001  # learning rate for Critic, not used by adadealta
beta1 = 0.5  # beta1 for adam. default=0.5
display_interval = 100

def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts

def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def print_params(net):
    # Print parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')



def set_train(m, train, fine_tune=False):
    if fine_tune:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
        if train:
            for p in m.classifier.parameters():
                p.requires_grad = True
            m.classifier.train()
    else:
        for p in m.parameters():
            p.requires_grad = True
        m.train()
    return m



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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

class Averager(object):
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


def imshow(img):
    img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))[:,:, [2,1,0]])
    plt.show()
