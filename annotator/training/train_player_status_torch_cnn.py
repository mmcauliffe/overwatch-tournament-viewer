import os
import shutil
import torch
import numpy as np

import torch.optim as optim
import random
from annotator.datasets.cnn_dataset import StatusCNNHDF5Dataset, CNNHDF5Dataset
from annotator.datasets.helper import randomSequentialSampler
import torch.nn as nn
from annotator.models.cnn import StatusCNN
from annotator.training.cnn_helper import train_model
from annotator.training.helper import Averager, load_set

TEST = True
train_dir = r'N:\Data\Overwatch\training_data\player_status'

cuda = True
seed = 1
batch_size = 600
test_batch_size = 1200
num_epochs = 20
early_stopping_threshold = 2
lr = 0.001 # learning rate for Critic, not used by adadealta
momentum = 0.5
beta1 = 0.5 # beta1 for adam. default=0.5
use_adam = True # whether to use adam (default is rmsprop)
use_adadelta = False # whether to use adadelta (default is rmsprop)
log_interval = 10
image_height = 72
image_width = 72
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

working_dir = r'N:\Data\Overwatch\models\player_status_test'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')
model_path = os.path.join(working_dir, 'model.pth')

spec_modes = [
    'original',
    'overwatch league',
    'overwatch league season 3',
    'world cup',
    'contenders'
]

input_set_files = {
    # 'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
}

set_files = {
    'hero': os.path.join(train_dir, 'hero_set.txt'),
    'alive': os.path.join(train_dir, 'alive_set.txt'),
    'ult': os.path.join(train_dir, 'ult_set.txt'),
    'status': os.path.join(train_dir, 'status_set.txt'),
    'antiheal': os.path.join(train_dir, 'antiheal_set.txt'),
    'immortal': os.path.join(train_dir, 'immortal_set.txt'),
    'nanoboosted': os.path.join(train_dir, 'nanoboosted_set.txt'),
}


if __name__ == '__main__':
    train_model(working_dir, train_dir, StatusCNN, StatusCNNHDF5Dataset, set_files, spec_modes,
                input_set_files=input_set_files, batch_size=batch_size, test_batch_size=test_batch_size)



