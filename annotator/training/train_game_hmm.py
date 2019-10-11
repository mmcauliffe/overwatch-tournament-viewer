import numpy as np
import h5py
import os
from annotator.config import BOX_PARAMETERS
from annotator.game_values import SPECTATOR_MODES, HERO_SET
from annotator.utils import get_local_file, \
    get_local_path,  FileVideoStream, Empty, get_vod_path
from collections import defaultdict

working_dir = r'E:\Data\Overwatch\models\game'
h5_path = os.path.join(working_dir, 'hmm_probs.h5')
from annotator.api_requests import get_player_states, get_train_rounds, get_train_rounds_plus, get_train_vods, get_game_states
from annotator.training.helper import load_set

set_files = {
    'game': os.path.join(working_dir, 'game_set.txt'),
    'map': os.path.join(working_dir, 'map_set.txt'),
    # 'film_format': os.path.join(working_dir, 'film_format_set.txt'),
    'left_color': os.path.join(working_dir, 'left_color_set.txt'),
    'right_color': os.path.join(working_dir, 'right_color_set.txt'),
    'spectator_mode': os.path.join(working_dir, 'spectator_mode_set.txt'),
    'left': os.path.join(working_dir, 'left_set.txt'),
    'right': os.path.join(working_dir, 'right_set.txt'),
}

values = {}
for k, v in set_files.items():
    values[k] = load_set(v)


init_counts = {}
transition_counts = {}
for k, v in values.items():
    init_counts[k] ={h1: 1 for h1 in v if h1}
    transition_counts[k] = {h1:{h2: 5 for h2 in v if h2} for h1 in v if h1}


def normalize_init_counts(counts):
    total_init_count = sum(counts.values())
    probs = {k: v /total_init_count for k,v in counts.items()}
    return probs

def normalize_trans_counts(counts):
    return {k1: {k2: v2 / sum(v1.values()) for k2, v2 in v1.items()} for k1, v1 in counts.items()}


if __name__ == '__main__':
    vods = get_train_vods()
    for v in vods:
        print(v['id'])
        states = get_game_states(v['id'])
        for k in init_counts.keys():
            d = states[k]
            initial_value = d[0]['status']
            init_counts[k][initial_value] += 1
            for i, interval in enumerate(d):
                cur_value = interval['status']
                transition_counts[k][cur_value][cur_value] += (interval['end'] - interval['begin'])
                if i < len(d) - 1:
                    next_value = d[i+1]['status']
                    transition_counts[k][cur_value][next_value] += 1
    init_probs = {}
    trans_probs = {}
    for k, v in init_counts.items():
        init_probs[k] = normalize_init_counts(v)
        trans_probs[k] = normalize_trans_counts(transition_counts[k])
        print(k)
        print(init_probs[k])
        print(trans_probs[k])
    hdf5_file = h5py.File(h5_path, mode='w')
    for k, v in values.items():
        hdf5_file.create_dataset("{}_init".format(k), (len(v),), np.float)
        hdf5_file.create_dataset("{}_trans".format(k), (len(v),len(v)), np.float)
    for k, probs in init_probs.items():
        for h1, p in probs.items():
            hdf5_file["{}_init".format(k)][values[k].index(h1)] = p
    for k, probs in trans_probs.items():
        for h1, v in probs.items():
            for h2, p in v.items():
                hdf5_file["{}_trans".format(k)][values[k].index(h1), values[k].index(h2)] = p
    hdf5_file.close()
