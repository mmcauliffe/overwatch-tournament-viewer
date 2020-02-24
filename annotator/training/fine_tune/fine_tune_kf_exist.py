import os
import torch
import numpy as np
import random
from annotator.datasets.cnn_dataset import KFExistsCNNHDF5Dataset
from annotator.models.cnn import KillFeedCNN
from annotator.training.cnn_helper import fine_tune_model

working_dir = r'N:\Data\Overwatch\models\kill_feed_exists'
if not os.path.exists(working_dir):
    raise Exception('Train general model first!')

log_dir = os.path.join(working_dir, 'log')
TEST = True
train_dir = r'N:\Data\Overwatch\training_data\kill_feed_ctc_base'

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
image_height = 32
image_width = 296
num_channels = 3
num_hidden = 256
num_workers = 0
n_test_disp = 10
display_interval = 100
manualSeed = 1234 # reproduce experiemnt
random_sample = True

spec_modes = [
    'original',
    #'overwatch league',
    'overwatch league season 3',
    #'world cup',
    'contenders'
]

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

set_files = {
    'exist': os.path.join(train_dir, 'exist_set.txt'),
    'size': os.path.join(train_dir, 'size_set.txt'),
}
input_set_files = {
    # 'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
}


if __name__ == '__main__':
    for sp in spec_modes:
        fine_tune_model(working_dir, train_dir, KillFeedCNN, KFExistsCNNHDF5Dataset,
                        set_files, sp, input_set_files=input_set_files, batch_size=batch_size,
                        test_batch_size=test_batch_size)

