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


from annotator.data_generation.classes import PlayerStatusGenerator, PlayerOCRGenerator, KillFeedCTCGenerator, \
    MidStatusGenerator, PauseStatusGenerator, ReplayStatusGenerator, GameGenerator, SmallerWindowStatusGenerator, \
    PlayerLSTMGenerator

training_data_directory = r'E:\Data\Overwatch\training_data'

cnn_status_train_dir = os.path.join(training_data_directory, 'player_status_cnn')
ocr_status_train_dir = os.path.join(training_data_directory, 'player_status_ocr')
lstm_status_train_dir = os.path.join(training_data_directory, 'player_status_lstm')
cnn_mid_train_dir = os.path.join(training_data_directory, 'mid_cnn')
lstm_mid_train_dir = os.path.join(training_data_directory, 'mid_lstm')
cnn_kf_train_dir = os.path.join(training_data_directory, 'kf_cnn')
lstm_kf_train_dir = os.path.join(training_data_directory, 'kf_lstm')


def generate_data(rounds):
    from decimal import Decimal
    import time as timepackage
    generators = [MidStatusGenerator(),
                  KillFeedCTCGenerator(debug=False),
                  PlayerStatusGenerator(),
                  PlayerLSTMGenerator()
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
        for i, g in enumerate(generators):
            g.add_new_round_info(r)
            print(g.generate_data)
            if g.generate_data:
                process_round = True
        if not process_round:
            continue
        time_step = min(x.minimum_time_step for x in generators if x.generate_data)
        for beg, end in r['sequences']:
            print(beg, end)
            fvs = FileVideoStream(get_vod_path(r['stream_vod']), beg + r['begin'], end + r['begin'], time_step,
                                  real_begin=r['begin']).start()
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
                    elif frame_ind % (g.minimum_time_step / time_step) == 0:
                        g.process_frame(frame, time_point, frame_ind)
                    average_times[i] += (timepackage.time()-begin)/100

                if frame_ind % 100 == 0:
                    print('Frame: {}/{}'.format(frame_ind, num_frames))
                    print(g.process_index)
                    for i, g in enumerate(generators):
                        print('Average process frame time for {}:'.format(type(g).__name__), average_times[i])
                    average_times = [0 for _ in generators]
                frame_ind += 1

        for g in generators:
            g.cleanup_round()
        print('Finished in {} seconds!'.format(timepackage.time() - begin_time))
    #for i, g in enumerate(generators):
    #    g.cleanup()

def generate_data_for_pauses(rounds):
    import time as timepackage
    generators = [ReplayStatusGenerator(debug=True), PauseStatusGenerator(debug=True), SmallerWindowStatusGenerator(debug=True)]
    #for g in generators:
    #    g.calculate_map_size(rounds)
    #    g.instantiate_environment()
    #generators = [PauseStatusGenerator(debug=True)]

    average_times = [0 for _ in generators]
    for round_index, r in enumerate(rounds):
        if r['stream_vod'] is None:
            continue
        print("Processing round {} of {}".format(round_index, len(rounds)))
        print(r)
        print(r['spectator_mode'])
        begin_time = timepackage.time()
        process_round = False
        for i, g in enumerate(generators):
            g.add_new_round_info(r)
            print(g.generate_data)
            if g.generate_data:
                process_round = True
        if not process_round:
            continue
        time_step = min(x.time_step for x in generators if x.generate_data)
        beg, end = r['begin'], r['end']
        print(beg, end)
        fvs = FileVideoStream(get_vod_path(r['stream_vod']),r['begin'], r['end'], time_step,
                              real_begin=r['begin']).start()
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
                    print('Average process frame time for {}:'.format(type(g).__name__), average_times[i])
                average_times = [0 for _ in generators]
            frame_ind += 1

        for g in generators:
            g.cleanup_round()
        print('Finished in {} seconds!'.format(timepackage.time() - begin_time))


def generate_data_for_game_cnn(vods):

    import time as timepackage
    generators = [GameGenerator()]

    average_times = [0, 0, 0, 0]
    for round_index, v in enumerate(vods):
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
        time_step = min(x.time_step for x in generators if x.generate_data)
        fvs = FileVideoStream(get_vod_path(v), 0, 0, time_step,
                              real_begin=0).start()
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
                if time_step == g.time_step:
                    g.process_frame(frame, time_point, frame_ind)
                elif frame_ind % (g.time_step / time_step) == 0:
                    g.process_frame(frame, time_point, frame_ind)
                average_times[i] += (timepackage.time()-begin)/100

            if frame_ind % 100 == 0:
                print('Frame: {}/{}'.format(frame_ind, num_frames))
                for i, g in enumerate(generators):
                    print('Average process frame time for {}:'.format(type(g).__name__), average_times[i])
                average_times = [0, 0, 0, 0]
            frame_ind += 1

        for g in generators:
            g.cleanup_round()
        print('Finished in {} seconds!'.format(timepackage.time() - begin_time))

def generate_data_for_cnn(rounds, vods):
    # rounds = rounds[:2]
    #generate_data_for_pauses(rounds)
    generate_data(rounds)
    #generate_data_for_game_cnn(vods)


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

    vods = get_train_vods()#[:max_count]
    max_count = 2

    #FILTER
    rounds = get_train_rounds()
    #rounds = [r for r in rounds if r['stream_vod'] is not None and r['stream_vod']['film_format'] == 'A']
    #rounds = rounds[:max_count]
    #hero_times = get_hero_play_time(rounds)
    #analyze_missing_vods(rounds, vods)
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
    # rounds = get_example_rounds()
    generate_data_for_cnn(rounds, vods)


