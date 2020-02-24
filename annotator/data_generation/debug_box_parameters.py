import requests
import os
import numpy as np
import h5py
import random
import cv2
import shutil
from annotator.utils import get_local_file, get_duration, \
    get_local_path,  FileVideoStream, Empty, get_vod_path
from annotator.api_requests import get_round_states, get_train_rounds, get_train_rounds_plus, get_train_vods

from annotator.game_values import SPECTATOR_MODES
from annotator.config import BOX_PARAMETERS, offsets

from annotator.data_generation.classes import PlayerStatusGenerator, PlayerOCRGenerator, KillFeedCTCGenerator, \
    MidStatusGenerator

ROUND = 15440

ROUND_TIME = (2 * 60) + 34

working_dir = r'N:\Data\Overwatch\models'
kf_exists_model_dir = os.path.join(working_dir, 'kill_feed_exists')
kf_ctc_model_dir = os.path.join(working_dir, 'kill_feed_ctc_base')

def display_round(r):
    import time as timepackage
    generators = [PlayerStatusGenerator(),
                  #PlayerOCRGenerator(),
                  KillFeedCTCGenerator(
                      exists_model_directory=kf_exists_model_dir, kf_ctc_model_directory=kf_ctc_model_dir
                  ),
                  #MidStatusGenerator()
                  ]
    print(r)
    print(r['spectator_mode'])
    print(r['stream_vod']['film_format'])
    begin_time = timepackage.time()
    for g in generators:
        g.add_new_round_info(r)
        if isinstance(g, KillFeedCTCGenerator):
            g.set_up_models()
        g.get_data(r)
        g.figure_slot_params(r)
    time_steps = [int(x.time_step * 10) for x in generators]
    print(time_steps)
    time_step = round(np.gcd.reduce(time_steps) / 10, 1)
    time_step = 0.1
    actual_duration, mode = get_duration(r['stream_vod'])
    offset = None
    if r['stream_vod']['id'] in offsets:
        offset = offsets[r['stream_vod']['id']]
    print(offset)
    for beg, end in r['sequences']:
        beg += 0.1
        end -= 0.1
        print(beg, end)
        if end < ROUND_TIME:
            continue
        fvs = FileVideoStream(get_vod_path(r['stream_vod']), round(ROUND_TIME + r['begin'], 1), round(end + r['begin'], 1), time_step,
                              real_begin=r['begin'], actual_duration=actual_duration, mode=mode, offset=offset).start()
        timepackage.sleep(1)
        frame_ind = 0
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            frame = frame['frame']
            print('READ', time_point)
            for i, g in enumerate(generators):
                print(g, time_step, g.time_step, time_step == g.time_step, frame_ind, (g.time_step / time_step), frame_ind % (g.time_step / time_step))
                if time_step == g.time_step or frame_ind % (g.time_step / time_step) == 0:
                    g.display_current_frame(frame, time_point, frame_ind)
            frame_ind += 1
            if time_point >= ROUND_TIME:
                cv2.imshow('frame_{}'.format(time_point), frame)
                cv2.waitKey()

    print('Finished in {} seconds!'.format(timepackage.time() - begin_time))


if __name__ == '__main__':
    rounds = get_train_rounds(round=ROUND)
    for round_index, r in enumerate(rounds):
        if r['id'] != ROUND:
            continue
        print(list(r.keys()))
        print(r['sequences'])
        print(r['id'])
        print(r['stream_vod'])
        if r['stream_vod'] is None:
            continue
        #if r['stream_vod']['film_format'] != 'U':
        #    continue
        local_path = get_vod_path(r['stream_vod'])
        if not os.path.exists(local_path):
            if get_local_path(r) is not None:
                shutil.move(get_local_path(r), local_path)
            else:
                print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
        print("Processing round {} of {}".format(round_index, len(rounds)))
        display_round(r)