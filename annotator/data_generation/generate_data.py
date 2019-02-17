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


from annotator.data_generation.classes import PlayerStatusGenerator, PlayerOCRGenerator, KillFeedCTCGenerator, MidStatusGenerator, SequenceDataGenerator

training_data_directory = r'E:\Data\Overwatch\training_data'

cnn_status_train_dir = os.path.join(training_data_directory, 'player_status_cnn')
ocr_status_train_dir = os.path.join(training_data_directory, 'player_status_ocr')
lstm_status_train_dir = os.path.join(training_data_directory, 'player_status_lstm')
cnn_mid_train_dir = os.path.join(training_data_directory, 'mid_cnn')
lstm_mid_train_dir = os.path.join(training_data_directory, 'mid_lstm')
cnn_kf_train_dir = os.path.join(training_data_directory, 'kf_cnn')
lstm_kf_train_dir = os.path.join(training_data_directory, 'kf_lstm')


def generate_data(rounds):
    import time as timepackage
    generators = [PlayerStatusGenerator(), PlayerOCRGenerator(), KillFeedCTCGenerator(), MidStatusGenerator()]
    average_times = [0, 0, 0, 0]
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
            if g.generate_data:
                process_round = True
        if not process_round:
            continue
        time_step = min(x.time_step for x in generators if x.generate_data)
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
                        g.process_frame(frame, time_point)
                    elif frame_ind % (g.time_step / time_step) == 0:
                        g.process_frame(frame, time_point)
                    average_times[i] += (timepackage.time()-begin)/100

                if frame_ind % 100 == 0:
                    print('Frame: {}/{}'.format(frame_ind, num_frames))
                    for i, g in enumerate(generators):
                        print('Average process frame time for {}:'.format(type(g).__name__), average_times[i])
                    average_times = [0, 0, 0, 0]
                frame_ind += 1
            for g in generators:
                if isinstance(g, SequenceDataGenerator):
                    g.save_current_sequence()

        for g in generators:
            g.cleanup_round()
        print('Finished in {} seconds!'.format(timepackage.time() - begin_time))


def generate_data_for_replay_cnn(rounds):
    train_dir = os.path.join(training_data_directory, 'replay_cnn')
    import time as timepackage
    debug = False
    time_step = 0.1
    hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    os.makedirs(train_dir, exist_ok=True)
    if os.path.exists(hd5_path):
        print('skipping replay cnn data')
        return
    print('beginning replay cnn data')

    # calc params
    num_frames = 0
    states = {}
    for r in rounds:
        states[r['id']] = get_round_states(r['id'])
        if len(states[r['id']]['replays']) == 1:
            continue
        for s in states[r['id']]['replays']:
            if s['status'] == 'replay':
                expected_frame_count = int((s['end'] - s['begin']) / time_step)
                print(expected_frame_count)
                num_frames += (int(expected_frame_count) + 1) * 2

    na_lab = 'n/a'
    replay_set = [na_lab, 'not_replay', 'replay']
    spectator_modes = [na_lab] + SPECTATOR_MODES

    print(num_frames)
    indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train

    params = BOX_PARAMETERS['O']['REPLAY']

    train_shape = (num_train, params['HEIGHT'], params['WIDTH'], 3)
    val_shape = (num_val, params['HEIGHT'], params['WIDTH'], 3)

    hdf5_file = h5py.File(hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        hdf5_file.create_dataset("{}_spectator_mode".format(pre), (count,), np.uint8)
        hdf5_file.create_dataset("{}_label".format(pre), (count,), np.uint8)

    frame_ind = 0
    for r in rounds:
        if len(states[r['id']]['replays']) == 1:
            continue

        spec_mode = r['spectator_mode'].lower()
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['REPLAY']
        for s in states[r['id']]['replays']:
            if s['status'] == 'replay':
                duration = s['end'] - s['begin']
                beg = s['begin'] - duration
                end = s['end'] + duration
                print(beg, end)
                fvs = FileVideoStream(get_vod_path(r['stream_vod']), beg + r['begin'], end + r['begin'], time_step,
                                      real_begin=r['begin']).start()
                timepackage.sleep(1.0)
                while True:
                    try:
                        frame, time_point = fvs.read()
                    except Empty:
                        break
                    time_point = round(time_point, 1)
                    if frame_ind >= len(indexes):
                        print('ignoring')
                        frame_ind += 1
                        continue

                    index = indexes[frame_ind]
                    if index < num_train:
                        pre = 'train'
                    else:
                        pre = 'val'
                        index -= num_train
                    if frame_ind != 0 and (frame_ind) % 100 == 0 and frame_ind < num_train:
                        print('Train data: {}/{}'.format(frame_ind, num_train))
                    elif frame_ind != 0 and frame_ind % 100 == 0:
                        print('Validation data: {}/{}'.format(frame_ind - num_train, num_val))
                    lab = 'not_replay'
                    if s['begin'] <= time_point <= s['end']:
                        lab = 'replay'
                    x = params['X']
                    y = params['Y']
                    box = frame[y: y + params['HEIGHT'],
                          x: x + params['WIDTH']]
                    if debug and lab == 'replay':
                        cv2.imshow('frame', frame)
                        cv2.imshow('box', box)
                        cv2.waitKey(0)
                    hdf5_file["{}_img".format(pre)][index, ...] = box[None]
                    hdf5_file["{}_round".format(pre)][index] = r['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    hdf5_file["{}_spectator_mode".format(pre)][index] = spectator_modes.index(spec_mode)
                    hdf5_file["{}_label".format(pre)][index] = replay_set.index(lab)

    hdf5_file.close()
    with open(os.path.join(train_dir, 'replay_set.txt'), 'w', encoding='utf8') as f:
        for p in replay_set:
            f.write('{}\n'.format(p))
    with open(os.path.join(train_dir, 'spectator_mode_set.txt'), 'w', encoding='utf8') as f:
        for p in spectator_modes:
            f.write('{}\n'.format(p))


def generate_data_for_pause_cnn(rounds):
    train_dir = os.path.join(training_data_directory, 'pause_cnn')
    import time as timepackage
    debug = False
    time_step = 0.1
    hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    os.makedirs(train_dir, exist_ok=True)
    if os.path.exists(hd5_path):
        print('skipping pause cnn data')
        return
    print('beginning pause cnn data')

    # calc params
    num_frames = 0
    states = {}
    for r in rounds:
        states[r['id']] = get_round_states(r['id'])
        print(states[r['id']])
        if len(states[r['id']]['pauses']) == 1:
            continue
        for s in states[r['id']]['pauses']:
            if s['status'] == 'paused':
                expected_frame_count = int((s['end'] - s['begin']) / time_step)
                print(expected_frame_count)
                num_frames += (int(expected_frame_count) + 1) * 2
    na_lab = 'n/a'
    pause_set = [na_lab, 'not_paused', 'paused']
    spectator_modes = [na_lab] + SPECTATOR_MODES

    print(num_frames)
    indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train

    params = BOX_PARAMETERS['O']['PAUSE']

    train_shape = (num_train, params['HEIGHT'], params['WIDTH'], 3)
    val_shape = (num_val, params['HEIGHT'], params['WIDTH'], 3)

    hdf5_file = h5py.File(hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        hdf5_file.create_dataset("{}_spectator_mode".format(pre), (count,), np.uint8)
        hdf5_file.create_dataset("{}_label".format(pre), (count,), np.uint8)

    frame_ind = 0
    for r in rounds:
        if len(states[r['id']]['pauses']) == 1:
            continue
        spec_mode = r['spectator_mode'].lower()
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['PAUSE']
        for s in states[r['id']]['pauses']:
            if s['status'] == 'paused':
                duration = s['end'] - s['begin']
                beg = s['begin'] - duration
                end = s['end'] + duration
                print(beg, end)
                fvs = FileVideoStream(get_vod_path(r['stream_vod']), beg + r['begin'], end + r['begin'], time_step,
                                      real_begin=r['begin']).start()
                timepackage.sleep(1.0)
                while True:
                    try:
                        frame, time_point = fvs.read()
                    except Empty:
                        break
                    time_point = round(time_point, 1)
                    if frame_ind >= len(indexes):
                        print('ignoring')
                        frame_ind += 1
                        continue

                    index = indexes[frame_ind]
                    if index < num_train:
                        pre = 'train'
                    else:
                        pre = 'val'
                        index -= num_train
                    if frame_ind != 0 and (frame_ind) % 100 == 0 and frame_ind < num_train:
                        print('Train data: {}/{}'.format(frame_ind, num_train))
                    elif frame_ind != 0 and frame_ind % 100 == 0:
                        print('Validation data: {}/{}'.format(frame_ind - num_train, num_val))
                    lab = 'not_paused'
                    if s['begin'] <= time_point <= s['end']:
                        lab = 'paused'
                    x = params['X']
                    y = params['Y']
                    box = frame[y: y + params['HEIGHT'],
                          x: x + params['WIDTH']]
                    if debug and lab == 'paused':
                        cv2.imshow('frame', frame)
                        cv2.imshow('box', box)
                        cv2.waitKey(0)
                    hdf5_file["{}_img".format(pre)][index, ...] = box[None]
                    hdf5_file["{}_round".format(pre)][index] = r['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    hdf5_file["{}_spectator_mode".format(pre)][index] = spectator_modes.index(spec_mode)
                    hdf5_file["{}_label".format(pre)][index] = pause_set.index(lab)

    hdf5_file.close()
    with open(os.path.join(train_dir, 'pause_set.txt'), 'w', encoding='utf8') as f:
        for p in pause_set:
            f.write('{}\n'.format(p))
    with open(os.path.join(train_dir, 'spectator_mode_set.txt'), 'w', encoding='utf8') as f:
        for p in spectator_modes:
            f.write('{}\n'.format(p))


def generate_data_for_game_cnn(vods):
    train_dir = os.path.join(training_data_directory, 'game_cnn')
    import time as timepackage
    debug = False
    time_step = 1
    os.makedirs(train_dir, exist_ok=True)
    generated_vods_path = os.path.join(train_dir, 'vods.txt')
    analyzed_vods = []
    labels = ['not_in_game', 'game']
    if os.path.exists(generated_vods_path):
        with open(generated_vods_path, 'r') as f:
            for line in f:
                analyzed_vods.append(int(line.strip()))
    hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    with open(os.path.join(train_dir, 'labels.txt'), 'w', encoding='utf8') as f:
        for p in labels:
            f.write('{}\n'.format(p))

    print('beginning game cnn data')
    error_set = []
    frames_per_seq = 100
    num_variations = 5
    resize_factor = 0.5
    seqs = {}
    print('analyzed', analyzed_vods)
    for v_i, v in enumerate(vods):
        print(v_i, len(vods))
        print(v)
        if v['id'] in analyzed_vods:
            continue
        analyzed_vods.append(v['id'])
        seqs[v['id']] = []
        cap = cv2.VideoCapture(get_vod_path(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur = num_frames / fps
        cap.release()
        num_sequences = 0
        num_frames = 0
        for i, s in enumerate(v['sequences']):
            beg, end = s
            seq_dur = end - beg
            non_seq_dur = seq_dur / 2
            prev_beg = beg - non_seq_dur
            if i == 0 and prev_beg < 0:
                prev_beg = 0
            elif i > 0 and prev_beg < v['sequences'][i - 1][1]:
                prev_beg = v['sequences'][i - 1][1]
            prev_dur = beg - prev_beg
            foll_end = end + non_seq_dur
            if i == len(v['sequences']) - 1 and foll_end > dur:
                foll_end = dur
            elif i < len(v['sequences']) - 1 and foll_end > v['sequences'][i + 1][0]:
                foll_end = v['sequences'][i + 1][0]
            foll_dur = foll_end - end
            expected_frame_count = int((seq_dur + prev_dur + foll_dur) / time_step)

            num_frames += (int(expected_frame_count) + 1)
            num_sequences += (int(expected_frame_count / frames_per_seq) + 1)
            seqs[v['id']].append([prev_beg, foll_end])
        num_sequences *= num_variations
        print(num_sequences)
        indexes = random.sample(range(num_sequences), num_sequences)
        num_train = int(num_sequences * 0.8)
        num_val = num_sequences - num_train

        params = BOX_PARAMETERS['O']['MID']
        train_shape = (
            num_train, frames_per_seq, int(params['HEIGHT'] * resize_factor), int(params['WIDTH'] * resize_factor), 3)
        val_shape = (
            num_val, frames_per_seq, int(params['HEIGHT'] * resize_factor), int(params['WIDTH'] * resize_factor), 3)

        if not os.path.exists(hd5_path):
            prev_train = 0
            prev_val = 0
            hdf5_file = h5py.File(hd5_path, mode='w')
            hdf5_file.create_dataset("train_mean", train_shape[1:], np.uint8)
            for pre in ['train', 'val']:
                count = num_train
                shape = train_shape
                if pre == 'val':
                    count = num_val
                    shape = val_shape
                hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8,
                                         maxshape=(None, shape[1], shape[2], shape[3], shape[4]))
                hdf5_file.create_dataset("{}_vod".format(pre), (count,), np.uint32, maxshape=(None,))
                hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32, maxshape=(None,))
                hdf5_file.create_dataset("{}_label".format(pre), (count, frames_per_seq), np.uint8,
                                         maxshape=(None, frames_per_seq))
                # hdf5_file.create_dataset("{}_prev_label".format(pre), (count,), np.uint8)
        else:
            hdf5_file = h5py.File(hd5_path, mode='a')
            prev_train = hdf5_file['train_img'].shape[0]
            prev_val = hdf5_file['val_img'].shape[0]
            for pre in ['train', 'val']:
                if pre == 'train':
                    new_count = prev_train + num_train
                else:
                    new_count = prev_val + num_val
                hdf5_file["{}_img".format(pre)].resize(new_count, axis=0)
                hdf5_file["{}_vod".format(pre)].resize(new_count, axis=0)
                hdf5_file["{}_time_point".format(pre)].resize(new_count, axis=0)
                hdf5_file["{}_label".format(pre)].resize(new_count, axis=0)

        print(num_frames, num_train, num_val)
        sequence_ind = 0
        for seq in seqs[v['id']]:
            beg, end = seq
            print(beg, end)
            fvs = FileVideoStream(get_vod_path(v), beg, end, time_step, real_begin=0).start()
            timepackage.sleep(1.0)
            # prev_label = labels[0]
            begin_time = timepackage.time()
            data = np.zeros((frames_per_seq,), dtype=np.uint8)

            variation_set = [(0, 0)]

            while len(variation_set) < num_variations:
                x_offset = random.randint(-5, 5)
                y_offset = random.randint(-5, 5)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))

            images = np.zeros((num_variations, frames_per_seq, int(params['HEIGHT'] * resize_factor),
                               int(params['WIDTH'] * resize_factor), 3),
                              dtype=np.uint8)

            j = 0
            while True:
                try:
                    frame, time_point = fvs.read()
                except Empty:
                    break
                if sequence_ind >= len(indexes):
                    print('ignoring')
                    sequence_ind += 1
                    continue
                time_point = round(time_point, 1)
                lab = labels[0]
                for s in v['sequences']:
                    if s[0] - 2 <= time_point <= s[1] + 2:
                        lab = labels[1]

                data[j] = labels.index(lab)
                x = params['X']
                y = params['Y']

                for i, (x_offset, y_offset) in enumerate(variation_set):
                    box = frame[y + y_offset: y + params['HEIGHT'] + y_offset,
                          x + x_offset: x + params['WIDTH'] + x_offset]
                    box = cv2.resize(box, (0, 0), fx=resize_factor, fy=resize_factor)
                    images[i, j, ...] = box[None]

                j += 1
                if j == frames_per_seq:
                    for i in range(num_variations):
                        index = indexes[sequence_ind]
                        if index < num_train:
                            pre = 'train'
                            index += prev_train
                        else:
                            pre = 'val'
                            index -= num_train
                            index += prev_val
                        print(sequence_ind, num_sequences)
                        if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                            print('Train data: {}/{}'.format(sequence_ind, num_train))
                        elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                            print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                        hdf5_file["{}_img".format(pre)][index, ...] = images[i, ...]

                        hdf5_file["{}_vod".format(pre)][index] = v['id']
                        hdf5_file["{}_time_point".format(pre)][index] = time_point
                        hdf5_file['{}_label'.format(pre)][index, ...] = data[None]
                        # hdf5_file['{}_prev_label'.format(pre)][index] = labels.index(prev_label)
                        # prev_label = labels[data[-1]]
                        sequence_ind += 1

                    variation_set = [(0, 0)]

                    while len(variation_set) < num_variations:
                        x_offset = random.randint(-5, 5)
                        y_offset = random.randint(-5, 5)
                        if (x_offset, y_offset) in variation_set:
                            continue
                        variation_set.append((x_offset, y_offset))

                    images = np.zeros((num_variations, frames_per_seq, int(params['HEIGHT'] * resize_factor),
                                       int(params['WIDTH'] * resize_factor), 3),
                                      dtype=np.uint8)

                    j = 0
            if j > 0:
                for i in range(num_variations):
                    index = indexes[sequence_ind]
                    if index < num_train:
                        pre = 'train'
                        index += prev_train
                    else:
                        pre = 'val'
                        index -= num_train
                        index += prev_val
                    print(sequence_ind, num_sequences)
                    if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                        print('Train data: {}/{}'.format(sequence_ind, num_train))
                    elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                        print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                    hdf5_file["{}_img".format(pre)][index, ...] = images[i, ...]

                    hdf5_file["{}_vod".format(pre)][index] = v['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    hdf5_file['{}_label'.format(pre)][index, ...] = data[None]
                    # hdf5_file['{}_prev_label'.format(pre)][index] = labels.index(prev_label)

                    sequence_ind += 1
            print('main loop took', timepackage.time() - begin_time)

        hdf5_file.close()
        with open(generated_vods_path, 'w') as f:
            for r in analyzed_vods:
                f.write('{}\n'.format(r))


def generate_data_for_cnn(rounds, vods, rounds_plus):
    # rounds = rounds[:2]
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


if __name__ == '__main__':
    rounds_plus = get_train_rounds_plus()
    max_count = 100
    rounds = get_train_rounds()#[:max_count]
    hero_times = get_hero_play_time(rounds)
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

    vods = get_train_vods()#[:max_count]
    # rounds = get_example_rounds()
    generate_data_for_cnn(rounds, vods, rounds_plus)
