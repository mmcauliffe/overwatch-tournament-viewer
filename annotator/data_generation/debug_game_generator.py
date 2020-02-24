import requests
import os
import numpy as np
import h5py
import random
import cv2
import shutil
from annotator.utils import get_local_file, get_duration, \
    get_local_path,  FileVideoStream, Empty, get_vod_path
from annotator.api_requests import get_round_states, get_train_rounds, get_train_rounds_plus, get_train_vods, get_vods

from annotator.game_values import SPECTATOR_MODES
from annotator.config import BOX_PARAMETERS, offsets

from annotator.data_generation.classes import GameGenerator

VOD = 3639

VOD_TIME = (15 * 60) + 0.2


def display_vod(v):
    import time as timepackage
    generators = [GameGenerator()]
    print(v)
    begin_time = timepackage.time()
    for g in generators:
        g.add_new_round_info(v)
        g.get_data(v)
        g.figure_slot_params(v)
    time_step = min(x.minimum_time_step for x in generators)
    actual_duration, mode = get_duration(v)
    offset = None
    if v['id'] in offsets:
        offset = offsets[v['id']]
    fvs = FileVideoStream(get_vod_path(v), VOD_TIME, 0, generators[0].broadcast_event_time_step,
                          real_begin=0, special_time_steps=generators[0].special_time_steps,
                          actual_duration=actual_duration, offset=offset, mode=mode
                          ).start()
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
            g.display_current_frame(frame, time_point, frame_ind)
        frame_ind += 1
        if time_point >= VOD_TIME:
            cv2.imshow('frame_{}'.format(time_point), frame)
            cv2.waitKey()

    print('Finished in {} seconds!'.format(timepackage.time() - begin_time))


if __name__ == '__main__':
    vods = get_vods(VOD)
    for round_index, v in enumerate(vods):
        #if r['stream_vod']['film_format'] != 'U':
        #    continue
        print(v)
        local_path = get_vod_path(v)
        if not os.path.exists(local_path):
            if get_local_path(v) is not None:
                shutil.move(get_local_path(v), local_path)
        print("Processing round {} of {}".format(round_index, len(vods)))
        display_vod(v)