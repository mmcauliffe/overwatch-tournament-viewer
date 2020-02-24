import os
import shutil
import torch
import numpy as np
import cv2

import random
from annotator.datasets.cnn_dataset import GameCNNHDF5Dataset
from annotator.models.cnn import GameNormCNN as GameCNN
from annotator.training.cnn_helper import test_errors

TEST = True
train_dir = r'N:\Data\Overwatch\training_data\game'

cuda = True
seed = 1
batch_size = 100
image_height = 144
image_width = 256
num_channels = 3

manualSeed = 1234 # reproduce experiemnt

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)



spec_modes = [
    #'original',
    #'overwatch league',
    'overwatch league season 3',
    #'world cup',
    #'contenders'
]

if __name__ == '__main__':

    model_dir = r'N:\Data\Overwatch\models\game_test'
    working_dir = r'N:\Data\Overwatch\models\debug\game'
    shutil.rmtree(working_dir, ignore_errors=True)
    os.makedirs(working_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')
    set_files = {
        'game': os.path.join(train_dir, 'game_set.txt'),
        'map': os.path.join(train_dir, 'map_set.txt'),
        'submap': os.path.join(train_dir, 'submap_set.txt'),
        'left_color': os.path.join(train_dir, 'left_color_set.txt'),
        'right_color': os.path.join(train_dir, 'right_color_set.txt'),
        'left': os.path.join(train_dir, 'left_set.txt'),
        'right': os.path.join(train_dir, 'right_set.txt'),
        'attacking_side': os.path.join(train_dir, 'attacking_side_set.txt'),
    }


    input_set_files = {
        #'film_format': os.path.join(train_dir, 'film_format_set.txt'),
        #'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
    }
    for sp in spec_modes:
        test_errors(model_dir, working_dir, train_dir, GameCNN, GameCNNHDF5Dataset, set_files,
                    input_set_files=input_set_files, batch_size=batch_size, spec_mode=sp)