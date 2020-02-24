import os
import shutil
import torch
import numpy as np
import random
from annotator.datasets.ctc_dataset import KillFeedDataset
from annotator.models import crnn
from annotator.training.helper import load_set
from annotator.training.ctc_helper import test_errors

TEST = True
train_dir = r'N:\Data\Overwatch\training_data\kill_feed_ctc_base'

cuda = True
seed = 1
batch_size = 100
image_height = 64
image_width = 64
num_channels = 3

spec_modes = [
    #'original',
    #'overwatch league',
    'overwatch league season 3',
    #'world cup',
    #'contenders'
]

manualSeed = 1234 # reproduce experiemnt

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

if __name__ == '__main__':
    model_dir = r'N:\Data\Overwatch\models\kill_feed_ctc_base'
    working_dir = r'N:\Data\Overwatch\models\debug\kill_feed_base'
    shutil.rmtree(working_dir, ignore_errors=True)
    os.makedirs(working_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'model.pth')

    label_path = os.path.join(train_dir, 'labels_set.txt')
    label_set = load_set(label_path)
    for sp in spec_modes:
        test_errors(model_dir, working_dir, train_dir, crnn.SideKillFeedCRNN, KillFeedDataset,
                    label_set, spec_mode=sp,
                    batch_size=batch_size, image_height=image_height, image_width=image_width, kill_feed=True)
