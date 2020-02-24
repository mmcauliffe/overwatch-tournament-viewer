import os
import torch
import numpy as np
import random
from annotator.datasets.cnn_dataset import CNNHDF5Dataset
from annotator.training.cnn_helper import train_model
from annotator.models.cnn import MidCNN

working_dir = r'N:\Data\Overwatch\models\mid'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')
TEST = True
train_dir = r'N:\Data\Overwatch\training_data\mid'

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


spec_modes = [
    'original',
    'overwatch league',
    'world cup',
    'contenders']

if __name__ == '__main__':
    input_set_files = {
        #'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
        'map': os.path.join(train_dir, 'map_set.txt'),
        'attacking_side': os.path.join(train_dir, 'attacking_side_set.txt'),
    }
    set_files = {
        'overtime': os.path.join(train_dir, 'overtime_set.txt'),
        'point_status': os.path.join(train_dir, 'point_status_set.txt'),
    }

    train_model(working_dir, train_dir, MidCNN, CNNHDF5Dataset, set_files, spec_modes,
                input_set_files=input_set_files, batch_size=batch_size, test_batch_size=test_batch_size)