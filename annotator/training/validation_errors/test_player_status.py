import os
import shutil
import torch
import numpy as np
import cv2

import random
from annotator.datasets.cnn_dataset import StatusCNNHDF5Dataset
from annotator.models.cnn import StatusCNN
from annotator.training.cnn_helper import test_errors

TEST = True
train_dir = r'N:\Data\Overwatch\training_data\player_status'

cuda = True
seed = 1
batch_size = 1200
image_height = 64
image_width = 64
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

    model_dir = r'N:\Data\Overwatch\models\player_status_test'
    working_dir = r'N:\Data\Overwatch\models\debug\player_status_test'
    shutil.rmtree(working_dir, ignore_errors=True)
    os.makedirs(working_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')

    input_set_files = {
        #'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
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

    for sp in spec_modes:
        test_errors(model_dir, working_dir, train_dir, StatusCNN, StatusCNNHDF5Dataset, set_files, sp,
                    input_set_files=input_set_files, batch_size=batch_size)
