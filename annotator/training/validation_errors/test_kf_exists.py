import os
import shutil
import torch
import numpy as np

import random
from annotator.datasets.cnn_dataset import KFExistsCNNHDF5Dataset
from annotator.models.cnn import KillFeedCNN
from annotator.training.cnn_helper import test_errors

train_dir = r'N:\Data\Overwatch\training_data\kill_feed_ctc_base'

batch_size = 1200

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

    model_dir = r'N:\Data\Overwatch\models\kill_feed_exists'
    working_dir = r'N:\Data\Overwatch\models\debug\kill_feed_exists'
    shutil.rmtree(working_dir, ignore_errors=True)
    os.makedirs(working_dir, exist_ok=True)

    input_set_files = {
        #'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
         }

    set_files = {
    'exist': os.path.join(train_dir, 'exist_set.txt'),
    'size': os.path.join(train_dir, 'size_set.txt'),
    }
    for sp in spec_modes:
        test_errors(model_dir, working_dir, train_dir, KillFeedCNN, KFExistsCNNHDF5Dataset,set_files, sp,
                    input_set_files=input_set_files, batch_size=batch_size)
