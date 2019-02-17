import requests
import os
import numpy as np
import h5py
import random
import cv2
import shutil
from annotator.config import BOX_PARAMETERS
from annotator.game_values import SPECTATOR_MODES
from annotator.utils import get_local_file, \
    get_local_path,  FileVideoStream, Empty, get_vod_path

from annotator.api_requests import get_round_states, get_train_rounds, get_train_rounds_plus, get_train_vods

training_data_directory = r'E:\Data\Overwatch\training_data'

status_train_dir = os.path.join(training_data_directory, 'player_status')

rounds = get_train_rounds()

with open(os.path.join(status_train_dir, 'spectator_modes.txt'), 'w') as f:
    f.write('\n'.join(SPECTATOR_MODES))

print(SPECTATOR_MODES)

for r in rounds:
    print(r)
    spec_mode = r['spectator_mode'].lower()
    print(spec_mode, SPECTATOR_MODES.index(spec_mode))
    hd5_path = os.path.join(status_train_dir, '{}.hdf5'.format(r['id']))
    if os.path.exists(hd5_path):
        with h5py.File(hd5_path, 'r+') as h5f:
            print('train_spectator_mode' in h5f.keys())
            print(h5f.keys())
            num = h5f['train_round'].shape[0]
            val_num = h5f['val_round'].shape[0]
            if 'train_spectator_mode' not in h5f.keys():
                h5f.create_dataset("train_spectator_mode", (num,), np.uint8, maxshape=(None,))
                h5f["train_spectator_mode"][0:num] = SPECTATOR_MODES.index(spec_mode)
                h5f.create_dataset("val_spectator_mode", (val_num,), np.uint8, maxshape=(None,))
                h5f["val_spectator_mode"][0:val_num] = SPECTATOR_MODES.index(spec_mode)


