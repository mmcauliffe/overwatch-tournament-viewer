import requests
import os
import numpy as np
import h5py
import random
import cv2
import shutil
from annotator.config import BOX_PARAMETERS, BASE_TIME_STEP, offsets
from annotator.game_values import SPECTATOR_MODES
from annotator.utils import get_local_file, get_duration,\
    get_local_path,  FileVideoStream, Empty, get_vod_path

from annotator.api_requests import get_round_states, get_train_rounds, get_train_rounds_plus, get_train_vods


from annotator.data_generation.classes import PlayerStatusGenerator, PlayerOCRGenerator, KillFeedCTCGenerator, \
    MidStatusGenerator, GameGenerator, PlayerLSTMGenerator

training_data_directory = r'N:\Data\Overwatch\training_data'

cnn_status_train_dir = os.path.join(training_data_directory, 'player_status_cnn')
ocr_status_train_dir = os.path.join(training_data_directory, 'player_status_ocr')
lstm_status_train_dir = os.path.join(training_data_directory, 'player_status_lstm')
cnn_mid_train_dir = os.path.join(training_data_directory, 'mid_cnn')
lstm_mid_train_dir = os.path.join(training_data_directory, 'mid_lstm')
cnn_kf_train_dir = os.path.join(training_data_directory, 'kf_cnn')
lstm_kf_train_dir = os.path.join(training_data_directory, 'kf_lstm')

DEFAULT_MODELS = False

spectator_modes = [
    'C',
    'O',
    '3'
]

working_dir = r'N:\Data\Overwatch\models'
if DEFAULT_MODELS:
    kf_exists_model_dir = None
    kf_ctc_model_dir = None
else:
    kf_exists_model_dir = os.path.join(working_dir, 'kill_feed_exists')
    kf_ctc_model_dir = os.path.join(working_dir, 'kill_feed_ctc_base')


def generate_data(rounds):
    from decimal import Decimal
    import time as timepackage
    generators = [MidStatusGenerator(),
                  KillFeedCTCGenerator(debug=False, exists_model_directory=kf_exists_model_dir,
                                       kf_ctc_model_directory=kf_ctc_model_dir
                                       ),
                  PlayerStatusGenerator(),
                  ]
    #for g in generators:
    #    g.calculate_map_size(rounds)
    #    g.instantiate_environment()

    average_times = [0 for _ in generators]
    for round_index, r in enumerate(rounds):
        if r['stream_vod'] is None:
            continue
        print("Processing round {} of {}".format(round_index, len(rounds)))
        print(r)
        print(r['spectator_mode'])
        begin_time = timepackage.time()
        process_round = False
        reset = False
        #if r['stream_vod']['film_format'] == 'K':
        #    reset = True

        for i, g in enumerate(generators):
            g.add_new_round_info(r, reset=reset)
            print(g, g.generate_data)
            if g.generate_data:
                process_round = True
        if not process_round:
            continue
        time_steps = [int(x.time_step * 10) for x in generators if x.generate_data]
        time_step = round(np.gcd.reduce(time_steps) / 10, 1)
        actual_duration, mode = get_duration(r['stream_vod'])
        offset = None
        if r['stream_vod']['id'] in offsets:
            offset = offsets[r['stream_vod']['id']]
        for beg, end in r['sequences']:
            beg += 0.1
            beg = round(beg, 1)
            end -= 0.1
            end = round(end, 1)
            print(beg, end)
            if end <= beg:
                continue
            fvs = FileVideoStream(get_vod_path(r['stream_vod']), round(beg + r['begin'], 1), round(end + r['begin'], 1), time_step,
                                  real_begin=r['begin'], use_window=False, actual_duration=actual_duration, offset=offset).start()
            timepackage.sleep(1.0)
            frame_ind = 0
            num_frames = int((end - beg) / time_step)
            while True:
                try:
                    frame, time_point = fvs.read()
                except Empty:
                    break
                for i, g in enumerate(generators):
                    begin = timepackage.time()
                    if time_step == g.time_step:
                        g.process_frame(frame, time_point, frame_ind)
                    elif frame_ind % (g.time_step / time_step) == 0:
                        g.process_frame(frame, time_point, frame_ind)
                    average_times[i] += (timepackage.time()-begin)/100

                if frame_ind % 100 == 0:
                    print('Frame: {}/{}'.format(frame_ind, num_frames))
                    for i, g in enumerate(generators):
                        if not g.generate_data:
                            continue
                        print('Average process frame time for {}:'.format(type(g).__name__), average_times[i])
                    average_times = [0 for _ in generators]
                frame_ind += 1

        for g in generators:
            g.cleanup_round()
        print('Finished in {} seconds!'.format(timepackage.time() - begin_time))
    #for i, g in enumerate(generators):
    #    g.cleanup()

def generate_data_for_game_cnn(vods):

    import time as timepackage
    generators = [GameGenerator()]
    checked_vods = set()
    average_times = [0, 0, 0, 0]
    for round_index, v in enumerate(vods):
        if v['id'] in checked_vods:
            continue
        checked_vods.add(v['id'])
        print("Processing vod {} of {}".format(round_index, len(vods)))
        print(v)
        begin_time = timepackage.time()
        process_vod = False
        for i, g in enumerate(generators):
            g.add_new_round_info(v)
            if g.generate_data:
                process_vod = True
        if not process_vod:
            print('skipping!')
            continue
        time_step = min(x.minimum_time_step for x in generators if x.generate_data)
        short_time_steps = generators[0].short_time_steps
        fvs = FileVideoStream(get_vod_path(v), 0, 0, generators[0].secondary_time_step,
                              real_begin=0, short_time_steps=short_time_steps).start()
        timepackage.sleep(1.0)
        frame_ind = 0
        end = fvs.end
        num_frames = int(end / time_step)
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            for i, g in enumerate(generators):
                begin = timepackage.time()
                g.process_frame(frame, time_point, frame_ind)
                average_times[i] += (timepackage.time()-begin)/100

            if frame_ind % 100 == 0:
                print('Frame: {}/{}'.format(time_point, end))
                for i, g in enumerate(generators):
                    print('Average process frame time for {}:'.format(type(g).__name__), average_times[i])
                average_times = [0, 0, 0, 0]
            frame_ind += 1

        for g in generators:
            g.cleanup_round()
        print('Finished in {} seconds!'.format(timepackage.time() - begin_time))



def save_round_info(rounds):
    with open(os.path.join(training_data_directory, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write('{} {} {} {}\n'.format(r['id'], r['game']['match']['id'], r['game']['game_number'],
                                           r['round_number']))



def get_hero_play_time(rounds):
    from collections import Counter
    from annotator.api_requests import get_player_states
    play_time = Counter()
    for i, r in enumerate(rounds):
        print(i)
        states = get_player_states(r['id'])
        for side, v in states.items():
            for (ind, values) in v.items():
                for h in values['hero']:
                    play_time[h['hero']['name']] += h['end'] - h['begin']
    print(play_time)
    play_time = sorted(play_time.items(), key=lambda x: x[1])
    print(play_time[int(len(play_time)/2)])
    median_hero_play_time = play_time[int(len(play_time)/2)][1]
    for k,v in play_time:
        print(k, v, median_hero_play_time/v)

    error

def analyze_missing_vods(rounds, vods):
    used_vod_ids = []
    count = 0
    for r in rounds:
        if r['stream_vod'] is not None:
            used_vod_ids.append(r['stream_vod']['id'])
            continue
        print(r)
        count += 1
    print('missing vods for {} rounds'.format(count))
    error

if __name__ == '__main__':
    #rounds_plus = get_train_rounds_plus()

    max_count = 2

    #FILTER
    for sp in spectator_modes:
        rounds = get_train_rounds(spectator_mode=sp)
        for r in rounds:
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
                    get_local_file(r)
        save_round_info(rounds)

        generate_data(rounds)


