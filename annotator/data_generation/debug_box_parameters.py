import requests
import os
import numpy as np
import h5py
import random
import cv2
import shutil
from annotator.utils import get_local_file, \
    get_local_path,  FileVideoStream, Empty, get_vod_path
from annotator.api_requests import get_round_states, get_train_rounds, get_train_rounds_plus, get_train_vods

from annotator.game_values import SPECTATOR_MODES
from annotator.config import BOX_PARAMETERS

from annotator.data_generation.classes import PlayerStatusGenerator, PlayerOCRGenerator, KillFeedCTCGenerator, \
    MidStatusGenerator

ROUND = 8049

ROUND_TIME = 205


def display_round(r):
    import time as timepackage
    generators = [PlayerStatusGenerator(), PlayerOCRGenerator(), KillFeedCTCGenerator(), MidStatusGenerator()]
    print(r)
    print(r['spectator_mode'])
    begin_time = timepackage.time()
    for g in generators:
        g.get_data(r)
        g.figure_slot_params(r)
    time_step = min(x.time_step for x in generators)
    for beg, end in r['sequences']:
        print(beg, end)
        fvs = FileVideoStream(get_vod_path(r['stream_vod']), r['begin'] + ROUND_TIME, end + r['begin'], time_step,
                              real_begin=r['begin']).start()
        timepackage.sleep(1.0)
        frame_ind = 0
        num_frames = int((end - beg) / time_step)
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            cv2.imshow('frame', frame)
            for i, g in enumerate(generators):
                g.display_current_frame(frame, time_point)
            frame_ind += 1
            cv2.waitKey()

    print('Finished in {} seconds!'.format(timepackage.time() - begin_time))


if __name__ == '__main__':
    rounds = get_train_rounds()
    for round_index, r in enumerate(rounds):
        if r['id'] != ROUND:
            continue
        print(list(r.keys()))
        print(r['sequences'])
        print(r['id'])
        print(r['stream_vod'])
        if r['stream_vod'] is None:
            continue
        local_path = get_vod_path(r['stream_vod'])
        if not os.path.exists(local_path):
            if get_local_path(r) is not None:
                shutil.move(get_local_path(r), local_path)
            else:
                print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
        print("Processing round {} of {}".format(round_index, len(rounds)))
        display_round(r)