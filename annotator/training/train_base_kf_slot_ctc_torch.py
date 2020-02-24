import os
import torch
import shutil
import numpy as np
import random
from annotator.datasets.ctc_dataset import KillFeedDataset
from annotator.training.helper import load_set
from annotator.training.ctc_helper import train_model

working_dir = r'N:\Data\Overwatch\models\kill_feed_ctc_base'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')

TEST = True
train_dir = r'N:\Data\Overwatch\training_data\kill_feed_ctc_base'

# params

cuda = True
seed = 1
batch_size = 200
test_batch_size = 400
num_epochs = 50
early_stopping_threshold = 2
lr = 0.001  # learning rate for Critic, not used by adadealta
beta1 = 0.5  # beta1 for adam. default=0.5
use_adam = True  # whether to use adam (default is rmsprop)
use_adadelta = False  # whether to use adadelta (default is rmsprop)
momentum = 0.5
log_interval = 10
image_height = 32
image_width = 296
num_channels = 3
num_hidden = 256
num_workers = 0
n_test_disp = 10
display_interval = 100
manualSeed = 1234  # reproduce experiemnt
random_sample = True
recent = False

spec_modes = [
    'original',
    'overwatch league',
    'overwatch league season 3',
    'world cup',
    'contenders']


random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)



### TRAINING
if __name__ == '__main__':
    from annotator.models import crnn
    label_path = os.path.join(train_dir, 'labels_set.txt')
    label_set = load_set(label_path)
    shutil.copyfile(label_path, os.path.join(working_dir, 'labels_set.txt'))

    train_model(working_dir, train_dir, crnn.SideKillFeedCRNN, KillFeedDataset, label_set, spec_modes=spec_modes,
                early_stopping_threshold=early_stopping_threshold, batch_size=batch_size,
                test_batch_size=test_batch_size, num_epochs=num_epochs, image_height=image_height,
                image_width=image_width, kill_feed=True)

