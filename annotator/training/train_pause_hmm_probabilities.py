import numpy as np
import h5py
import os
from annotator.config import BOX_PARAMETERS
from annotator.game_values import SPECTATOR_MODES, HERO_SET
from annotator.utils import get_local_file, \
    get_local_path,  FileVideoStream, Empty, get_vod_path
from collections import defaultdict

directories = {'pause': r'E:\Data\Overwatch\models\pause',
               'replay': r'E:\Data\Overwatch\models\replay',
            #'smaller_window': r'E:\Data\Overwatch\models\smaller_window',
               }

hdf5_paths = {}
for k, w in directories.items():
    hdf5_paths[k] = os.path.join(w, 'hmm_probs.h5')
from annotator.api_requests import get_round_states, get_train_rounds, get_train_rounds_plus, get_train_vods
na_lab = 'n/a'
values = {
    'pause': ['not_pause', 'pause'],
    'replay': ['not_replay', 'replay'],
    #'smaller_window': ['not_smaller_window', 'smaller_window']
}

recent = True

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
    rounds = get_train_rounds()
    for r in rounds:
        print(r['id'])
        states = get_round_states(r['id'])
        for k in init_counts.keys():
            d = states[k]
            initial_value = d[0]['status']
            init_counts[k][initial_value] += 1
            for i, interval in enumerate(d):
                cur_value = interval['status']
                transition_counts[k][cur_value][cur_value] += (interval['end'] - interval['begin'])
                if i < len(d) - 1:
                    next_value = d[i+1]['status']
                    transition_counts[k][cur_value][next_value] += 10
    init_probs = {}
    trans_probs = {}
    for k, v in init_counts.items():
        init_probs[k] = normalize_init_counts(v)
        trans_probs[k] = normalize_trans_counts(transition_counts[k])
    hdf5_files = {}
    for k, p in hdf5_paths.items():
        hdf5_files[k] = h5py.File(p, mode='w')
    for k, v in values.items():
        hdf5_files[k].create_dataset("{}_init".format(k), (len(v),), np.float)
        hdf5_files[k].create_dataset("{}_trans".format(k), (len(v),len(v)), np.float)
    for k, probs in init_probs.items():
        print(k)
        for h1, p in probs.items():
            print('   ', h1, p)
            hdf5_files[k]["{}_init".format(k)][values[k].index(h1)] = p
    for k, probs in trans_probs.items():
        print(k)
        for h1, v in probs.items():
            for h2, p in v.items():
                print('   ', h1, h2, p)
                hdf5_files[k]["{}_trans".format(k)][values[k].index(h1), values[k].index(h2)] = p
    for k in hdf5_files.keys():
        hdf5_files[k].close()
