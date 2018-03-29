import requests
import os
import numpy as np
import h5py
import random
import cv2
import shutil
from annotator.utils import get_local_file, HERO_SET, ABILITY_SET, BOX_PARAMETERS, SPECTATOR_MODES, PLAYER_SET, \
    get_player_states, get_local_path, MAP_SET, MAP_MODE_SET, \
    get_kf_events, get_round_states, get_train_rounds, FileVideoStream, look_up_player_state, look_up_round_state, \
    load_set, Empty, calculate_ability_boundaries, calculate_first_hero_boundaries, calculate_hero_boundaries, \
    FileVideoStreamRange, get_event_ranges, COLOR_SET, calculate_first_player_boundaries, calculate_assist_boundaries, \
    get_vod_path, get_train_vods, get_train_rounds_plus, PLAYER_CHARACTER_SET

training_data_directory = r'E:\Data\Overwatch\training_data'

cnn_status_train_dir = os.path.join(training_data_directory, 'player_status_cnn')
lstm_status_train_dir = os.path.join(training_data_directory, 'player_status_lstm')
cnn_mid_train_dir = os.path.join(training_data_directory, 'mid_cnn')
lstm_mid_train_dir = os.path.join(training_data_directory, 'mid_lstm')
cnn_kf_train_dir = os.path.join(training_data_directory, 'kf_cnn')
lstm_kf_train_dir = os.path.join(training_data_directory, 'kf_lstm')


def generate_data_for_player_cnn(rounds):
    debug = False
    import time as timepackage
    os.makedirs(cnn_status_train_dir, exist_ok=True)
    status_hd5_path = os.path.join(cnn_status_train_dir, 'dataset.hdf5')
    ocr_hd5_path = os.path.join(cnn_status_train_dir, 'ocr_dataset.hdf5')
    with open(os.path.join(cnn_status_train_dir, 'characters.txt'), 'w', encoding='utf8') as f:
        for c in PLAYER_CHARACTER_SET:
            f.write('{}\n'.format(c))
    if os.path.exists(status_hd5_path):
        print('skipping player cnn data')
        return
    print('beginning player cnn data')
    time_step = 0.1
    frames_per_seq = 100
    frames = []
    na_lab = 'n/a'
    set_files = {'hero': os.path.join(cnn_status_train_dir, 'hero_set.txt'),
                 'ult': os.path.join(cnn_status_train_dir, 'ult_set.txt'),
                 'alive': os.path.join(cnn_status_train_dir, 'alive_set.txt'),
                 'spectator_mode': os.path.join(cnn_status_train_dir, 'spectator_mode_set.txt'), }
    end_set_files = {
                 'player': os.path.join(cnn_status_train_dir, 'player_set.txt'),
                 'color': os.path.join(cnn_status_train_dir, 'color_set.txt'),}
    sets = {'hero': [na_lab] + HERO_SET,
            'ult': [na_lab, 'no_ult', 'has_ult'],
            'alive': [na_lab, 'alive', 'dead'],
            'spectator_mode': [na_lab] + SPECTATOR_MODES}
    end_sets = {
            'player': [na_lab] + PLAYER_SET,
            'color': [na_lab] + COLOR_SET,}

    params = BOX_PARAMETERS['O']['LEFT']
    #rounds = rounds[:3]
    num_frames = 0
    num_sequences = 0
    error_set = {}
    with open(os.path.join(cnn_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write('{}\n'.format(r['id']))
            if debug:
                if r['game']['match']['film_format'] != 'W':
                    continue
                if error_set and r['id'] not in [x[0] for x in error_set]:
                    continue

            for beg, end in r['sequences']:
                print(beg, end)
                expected_frame_count = int((end - beg) / time_step)
                num_frames += (int(expected_frame_count) + 1) * 12
                num_sequences += (int(expected_frame_count / frames_per_seq) + 1) * 12
    print(num_sequences, num_frames)

    indexes = random.sample(range(num_sequences), num_sequences)
    ocr_indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_sequences * 0.8)
    num_train_frames = int(num_frames * 0.8)
    num_val = num_sequences - num_train
    num_val_frames = num_frames - num_train_frames
    print(num_train_frames, num_val_frames)
    train_shape = (num_train, frames_per_seq, params['HEIGHT'], params['WIDTH'], 3)
    train_ocr_shape = (num_train_frames, params['WIDTH'], 12)
    val_shape = (num_val, frames_per_seq, params['HEIGHT'], params['WIDTH'], 3)
    val_ocr_shape = (num_val_frames, params['WIDTH'], 12)

    hdf5_file = h5py.File(status_hd5_path, mode='w')
    ocr_hdf5_file =  h5py.File(ocr_hd5_path, mode='w')
    max_player_name_length = 12
    for pre in ['train', 'val']:
        if pre == 'train':
            shape = train_shape
            ocr_shape = train_ocr_shape
            num = num_train
            ocr_count = num_train_frames
        else:
            shape = val_shape
            ocr_shape = val_ocr_shape
            num = num_val
            ocr_count = num_val_frames
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (num,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (num,), np.float32)
        hdf5_file.create_dataset("{}_prev_hero_label".format(pre), (num,), np.uint8)
        hdf5_file.create_dataset("{}_prev_ult_label".format(pre), (num,), np.uint8)
        hdf5_file.create_dataset("{}_prev_alive_label".format(pre), (num,), np.uint8)
        for k in sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (num, frames_per_seq), np.uint8)
        for k in end_sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (num,), np.uint8)

        ocr_hdf5_file.create_dataset("{}_img".format(pre), ocr_shape, np.uint8)
        ocr_hdf5_file.create_dataset("{}_label_sequence".format(pre), (ocr_count, max_player_name_length), np.uint32)
        ocr_hdf5_file.create_dataset("{}_label_sequence_length".format(pre), (ocr_count,), np.uint8)
    # hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

    sequence_ind = 0
    sides = ['left', 'right']
    frame_ind = 0
    for round_index, r in enumerate(rounds):
        if debug:
            if r['game']['match']['film_format'] != 'W':
                continue
            if error_set and r['id'] not in [x[0] for x in error_set]:
                continue
            if error_set:
                debug_error = [x for x in error_set if r['id'] == x[0]][0]
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
        left_params = BOX_PARAMETERS[r['stream_vod']['film_format']]['LEFT']
        right_params = BOX_PARAMETERS[r['stream_vod']['film_format']]['RIGHT']
        states = get_player_states(r['id'])
        left_color = r['game']['left_team']['color'].lower()
        spec_mode = r['spectator_mode'].lower()
        right_color = r['game']['right_team']['color'].lower()

        for beg, end in r['sequences']:
            print(beg, end)
            fvs = FileVideoStream(get_vod_path(r['stream_vod']), beg + r['begin'], end + r['begin'], time_step,
                                  real_begin=r['begin']).start()
            timepackage.sleep(1.0)

            print('begin main loop')
            begin_time = timepackage.time()
            images = {}
            data = {}
            for side in sides:
                for i in range(6):
                    images[(side, i)] = np.zeros((frames_per_seq, left_params['HEIGHT'], left_params['WIDTH'], 3),
                                                 dtype=np.uint8)
                    data[(side, i)] = {}

                    for k, s in sets.items():
                        data[(side, i)][k] = np.zeros((frames_per_seq,), dtype=np.uint8)

                    for k, s in end_sets.items():
                        data[(side, i)][k] = 0
            j = 0
            prev_heroes = {}
            prev_ults = {}
            prev_alives = {}
            for side in ['left', 'right']:
                for player_ind in range(6):
                    prev_heroes[(side, player_ind)] = sets['hero'].index(na_lab)
                    prev_ults[(side, player_ind)] = sets['ult'].index(na_lab)
                    prev_alives[(side, player_ind)] = sets['alive'].index(na_lab)
            cur_heroes = {}
            cur_ults = {}
            cur_alives = {}
            for side in ['left', 'right']:
                for player_ind in range(6):
                    cur_heroes[(side, player_ind)] = sets['hero'].index(na_lab)
                    cur_ults[(side, player_ind)] = sets['ult'].index(na_lab)
                    cur_alives[(side, player_ind)] = sets['alive'].index(na_lab)
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
                # if debug and time_point < debug_error[1]:
                #    continue
                for side in ['left', 'right']:
                    if side == 'left':
                        params = left_params
                    else:
                        params = right_params
                    for player_ind in range(6):
                        print(frame_ind, len(ocr_indexes))
                        ocr_index = ocr_indexes[frame_ind]
                        if ocr_index < num_train_frames:
                            pre = 'train'
                        else:
                            pre = 'val'
                            ocr_index -= num_train_frames
                        d = look_up_player_state(side, player_ind, time_point, states)
                        d['spectator_mode'] = spec_mode
                        if side == 'left':
                            d['color'] = left_color
                        else:
                            d['color'] = right_color
                        cur_heroes[(side, player_ind)] = sets['hero'].index(d['hero'])
                        cur_ults[(side, player_ind)] = sets['ult'].index(d['ult'])
                        cur_alives[(side, player_ind)] = sets['alive'].index(d['alive'])
                        x = params['X']
                        y = params['Y']
                        x += (params['WIDTH'] + params['MARGIN']) * (player_ind)
                        box = frame[y: y + params['HEIGHT'],
                              x: x + params['WIDTH']]
                        name_box = box[34:46, :]
                        gray = cv2.cvtColor(name_box, cv2.COLOR_BGR2GRAY)
                        gray = cv2.threshold(gray, 0, 255,
                                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                        images[(side, player_ind)][j, ...] = box[None]
                        ocr_hdf5_file["{}_img".format(pre)][ocr_index, ...] = np.swapaxes(gray, 1, 0)[None]
                        player_name_length = len(d['player'])
                        ocr_hdf5_file["{}_label_sequence_length".format(pre)][ocr_index] = player_name_length
                        for i in range(max_player_name_length):
                            if i < player_name_length:
                                ocr_hdf5_file["{}_label_sequence".format(pre)][ocr_index, i] = PLAYER_CHARACTER_SET.index(d['player'][i])
                            else:
                                ocr_hdf5_file["{}_label_sequence".format(pre)][ocr_index, i] = len(PLAYER_CHARACTER_SET)
                        for k, s in sets.items():
                            data[(side, player_ind)][k][j] = sets[k].index(d[k])
                        for k, s in end_sets.items():
                            data[(side, player_ind)][k] = end_sets[k].index(d[k])
                        if debug:  # and side == 'right' and player_ind == 1:  # and sequence_ind % 25 == 0:
                            print(r['id'], time_point)
                            print(d)
                            cv2.imshow('frame_{}_{}'.format(side, player_ind), box)
                        if frame_ind < num_frames - 1:
                            frame_ind += 1
                if debug:
                    cv2.imshow('frame', frame)
                    cv2.waitKey(0)

                j += 1
                if j == frames_per_seq:
                    for side in ['left', 'right']:
                        for player_ind in range(6):
                            index = indexes[sequence_ind]
                            if index < num_train:
                                pre = 'train'
                            else:
                                pre = 'val'
                                index -= num_train
                            print(sequence_ind, num_sequences)
                            if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                                print('Train data: {}/{}'.format(sequence_ind, num_train))
                            elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                                print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                            hdf5_file["{}_img".format(pre)][index, ...] = images[(side, player_ind)][None]
                            images[(side, player_ind)] = np.zeros(
                                (frames_per_seq, left_params['HEIGHT'], left_params['WIDTH'], 3), dtype=np.uint8)
                            hdf5_file["{}_round".format(pre)][index] = r['id']
                            hdf5_file["{}_time_point".format(pre)][index] = time_point
                            hdf5_file["{}_prev_hero_label".format(pre)][index] = prev_heroes[(side, player_ind)]
                            hdf5_file["{}_prev_ult_label".format(pre)][index] = prev_ults[(side, player_ind)]
                            hdf5_file["{}_prev_alive_label".format(pre)][index] = prev_alives[(side, player_ind)]
                            for k, s in sets.items():
                                hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = data[(side, player_ind)][k][None]
                                data[(side, player_ind)][k] = np.zeros((frames_per_seq,), dtype=np.uint8)
                            for k, s in end_sets.items():
                                hdf5_file['{}_{}_label'.format(pre, k)][index] = data[(side, player_ind)][k]
                                data[(side, player_ind)][k] = 0
                            prev_heroes = cur_heroes
                            prev_ults = cur_ults
                            prev_alives = cur_alives
                            sequence_ind += 1
                    j = 0

                    # if pre == 'train':
                    #    mean += box / num_train
                # if (r['id'], time_point) in error_set:
                #    error
            if j > 0:
                for side in ['left', 'right']:
                    for player_ind in range(6):
                        index = indexes[sequence_ind]
                        if index < num_train:
                            pre = 'train'
                        else:
                            pre = 'val'
                            index -= num_train
                        if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                            print('Train data: {}/{}'.format(sequence_ind, num_train))
                        elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                            print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                        hdf5_file["{}_img".format(pre)][index, ...] = images[(side, player_ind)][None]
                        images[(side, player_ind)] = np.zeros(
                            (frames_per_seq, left_params['HEIGHT'], left_params['WIDTH'], 3), dtype=np.uint8)
                        hdf5_file["{}_round".format(pre)][index] = r['id']
                        hdf5_file["{}_time_point".format(pre)][index] = time_point
                        hdf5_file["{}_prev_hero_label".format(pre)][index] = prev_heroes[(side, player_ind)]
                        hdf5_file["{}_prev_ult_label".format(pre)][index] = prev_ults[(side, player_ind)]
                        hdf5_file["{}_prev_alive_label".format(pre)][index] = prev_alives[(side, player_ind)]
                        for k, s in sets.items():
                            hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = data[(side, player_ind)][k][None]
                            data[(side, player_ind)][k] = np.zeros((frames_per_seq,), dtype=np.uint8)
                        for k, s in end_sets.items():
                            hdf5_file['{}_{}_label'.format(pre, k)][index] = data[(side, player_ind)][k]
                        sequence_ind += 1

            print('main loop took', timepackage.time() - begin_time)

    hdf5_file.close()
    for k, v in set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            s = sets[k]
            for p in s:
                f.write('{}\n'.format(p))
    for k, v in end_set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            s = end_sets[k]
            for p in s:
                f.write('{}\n'.format(p))


def generate_data_for_mid_cnn(rounds):
    import time as timepackage
    debug = False
    time_step = 0.1
    status_hd5_path = os.path.join(cnn_mid_train_dir, 'dataset.hdf5')
    end_set_files = {'attacking_color': os.path.join(cnn_mid_train_dir, 'color_set.txt'),
                     'map': os.path.join(cnn_mid_train_dir, 'map_set.txt'),
                     'map_mode': os.path.join(cnn_mid_train_dir, 'map_mode_set.txt'),
                     'round_number': os.path.join(cnn_mid_train_dir, 'round_number_set.txt'),
                     'spectator_mode': os.path.join(cnn_mid_train_dir, 'spectator_mode_set.txt'), }
    set_files = {
        'overtime': os.path.join(cnn_mid_train_dir, 'overtime_set.txt'),
        'point_status': os.path.join(cnn_mid_train_dir, 'point_set.txt'),
    }
    na_lab = 'n/a'
    end_sets = {'attacking_color': [na_lab] + COLOR_SET,
                'map': [na_lab] + MAP_SET,
                'map_mode': [na_lab] + MAP_MODE_SET,
                'round_number': range(1, 7),
                'spectator_mode': [na_lab] + SPECTATOR_MODES}
    sets = {
        'overtime': [na_lab] + ['not_overtime', 'overtime'],
        'point_status': [na_lab] + sorted(['Assault_A', 'Assault_B',
                                           'Escort_1', 'Escort_2', 'Escort_3'] +
                                          ['Control_' + x for x in end_sets['attacking_color']]),
    }
    os.makedirs(cnn_mid_train_dir, exist_ok=True)
    if os.path.exists(status_hd5_path):
        print('skipping mid cnn data')
        return
    print('beginning mid cnn data')
    error_set = [(7718, 205)]
    # calc params
    num_frames = 0
    num_sequences = 0
    frames_per_seq = 100
    with open(os.path.join(cnn_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            if debug:
                if error_set and r['id'] not in [x[0] for x in error_set]:
                    continue
                debug_error = [x for x in error_set if x[0] == r['id']][0]

            for beg, end in r['sequences']:
                expected_frame_count = int((end - beg) / time_step)
                num_frames += (int(expected_frame_count) + 1)
                num_sequences += (int(expected_frame_count / frames_per_seq) + 1)
    print(num_sequences)
    indexes = random.sample(range(num_sequences), num_sequences)
    num_train = int(num_sequences * 0.8)
    num_val = num_sequences - num_train

    params = BOX_PARAMETERS['O']['MID']
    train_shape = (num_train, frames_per_seq, int(params['HEIGHT'] * 0.3), int(params['WIDTH'] * 0.3), 3)
    val_shape = (num_val, frames_per_seq, int(params['HEIGHT'] * 0.3), int(params['WIDTH'] * 0.3), 3)

    hdf5_file = h5py.File(status_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        for k in sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count, frames_per_seq), np.uint8)
        for k in end_sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count,), np.uint8)

    print(num_frames, num_train, num_val)
    sequence_ind = 0
    for i, r in enumerate(rounds):
        print(i, len(rounds))
        if debug:
            # if r['game']['match']['film_format'] != '2':
            #    continue
            if error_set and r['id'] not in [x[0] for x in error_set]:
                continue
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'],
              r['game']['match']['film_format'])
        params = BOX_PARAMETERS[r['game']['match']['film_format']]['MID']
        spec_mode = r['spectator_mode'].lower()

        states = get_round_states(r['id'])
        left_color = r['game']['left_team']['color'].lower()
        attacking_color = r['attacking_color'].lower()

        right_color = r['game']['right_team']['color'].lower()
        map = r['game']['map']['name'].lower()
        map_mode = r['game']['map']['mode'].lower()

        for beg, end in r['sequences']:
            print(beg, end)
            fvs = FileVideoStream(get_vod_path(r['stream_vod']), beg + r['begin'], end + r['begin'], time_step,
                                  real_begin=r['begin']).start()
            timepackage.sleep(1.0)
            print('begin main loop')
            begin_time = timepackage.time()
            data = {}
            for k, s in sets.items():
                data[k] = np.zeros((frames_per_seq,), dtype=np.uint8)
            end_data = {'attacking_color': attacking_color, 'map': map, 'spectator_mode': spec_mode,
                        'map_mode': map_mode, 'round_number': r['round_number']}
            images = np.zeros((frames_per_seq, int(params['HEIGHT'] * 0.3), int(params['WIDTH'] * 0.3), 3),
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
                d = look_up_round_state(time_point, states)
                for k, v in sets.items():
                    if k == 'point_status':
                        if d[k] == 'Control_none':
                            d[k] = 'Control_n/a'
                        elif d[k] == 'Control_left':
                            d[k] = 'Control_' + left_color
                        elif d[k] == 'Control_right':
                            d[k] = 'Control_' + right_color
                    data[k][j] = sets[k].index(d[k])
                x = params['X']
                box = frame[params['Y']: params['Y'] + params['HEIGHT'],
                      x: x + params['WIDTH']]
                box = cv2.resize(box, (0, 0), fx=0.3, fy=0.3)
                if debug and time_point >= debug_error[1]:
                    print(params)
                    print(time_point, d)
                    cv2.imshow('frame', frame)
                    cv2.imshow('box', frame[params['Y']: params['Y'] + params['HEIGHT'],
                                      x: x + params['WIDTH']])
                    cv2.imshow('box_resized', box)
                    cv2.waitKey(0)
                images[j, ...] = box[None]
                j += 1
                if j == frames_per_seq:
                    index = indexes[sequence_ind]
                    if index < num_train:
                        pre = 'train'
                    else:
                        pre = 'val'
                        index -= num_train
                    print(sequence_ind, num_sequences)
                    if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                        print('Train data: {}/{}'.format(sequence_ind, num_train))
                    elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                        print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                    hdf5_file["{}_img".format(pre)][index, ...] = images[None]

                    images = np.zeros(
                        (frames_per_seq, int(params['HEIGHT'] * 0.3), int(params['WIDTH'] * 0.3), 3), dtype=np.uint8)
                    hdf5_file["{}_round".format(pre)][index] = r['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    for k, s in sets.items():
                        hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = data[k][None]
                        data[k] = np.zeros((frames_per_seq,), dtype=np.uint8)
                    for k, s in end_sets.items():
                        hdf5_file['{}_{}_label'.format(pre, k)][index] = s.index(end_data[k])

                    sequence_ind += 1
                    j = 0
            if j > 0:
                index = indexes[sequence_ind]
                if index < num_train:
                    pre = 'train'
                else:
                    pre = 'val'
                    index -= num_train
                print(sequence_ind, num_sequences)
                if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                    print('Train data: {}/{}'.format(sequence_ind, num_train))
                elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                    print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                hdf5_file["{}_img".format(pre)][index, ...] = images[None]

                images = np.zeros(
                    (frames_per_seq, int(params['HEIGHT'] * 0.3), int(params['WIDTH'] * 0.3), 3), dtype=np.uint8)
                hdf5_file["{}_round".format(pre)][index] = r['id']
                hdf5_file["{}_time_point".format(pre)][index] = time_point
                for k, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = data[k][None]
                    data[k] = np.zeros((frames_per_seq,), dtype=np.uint8)
                for k, s in end_sets.items():
                    hdf5_file['{}_{}_label'.format(pre, k)][index] = s.index(end_data[k])

                sequence_ind += 1
            print('main loop took', timepackage.time() - begin_time)

    hdf5_file.close()
    for k, v in set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in sets[k]:
                f.write('{}\n'.format(p))
    for k, v in end_set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in end_sets[k]:
                f.write('{}\n'.format(p))


def construct_kf_at_time(events, time):
    window = 7.3
    possible_kf = []
    event_at_time = False
    for e in events:
        if e['time_point'] > time + 0.25:
            break
        elif e['time_point'] > time:
            possible_kf.insert(0, {'time_point': e['time_point'],
                                   'first_hero': 'n/a', 'first_color': 'n/a', 'ability': 'n/a', 'headshot': 'n/a',
                                   'second_hero': 'n/a',
                                   'second_color': 'n/a', })
        if time - window <= e['time_point'] <= time:
            if abs(time - e['time_point']) < 0.05:
                event_at_time = True
            for k, v in e.items():
                if isinstance(v, str):
                    e[k] = v.lower()
                # if 'color' in k:
                #    if e[k] != 'white':
                #        e[k] = 'nonwhite'
            possible_kf.append(e)
    possible_kf = sorted(possible_kf, key=lambda x: -1 * x['time_point'])
    return possible_kf[:6], event_at_time


def generate_negative_kf_examples(r, events, number):
    duration = int(r['end'] - r['begin'])
    possible = []
    for t in range(duration):
        kf = construct_kf_at_time(events, t)
        for i in range(6):
            if i < len(kf):
                continue
            possible.append(
                {'video_path': get_vod_path(r['stream_vod']), 'pos': i, 'time_point': t + r['begin'],
                 'first_hero': 'n/a', 'first_color': 'n/a', 'ability': 'n/a', 'headshot': 'n/a', 'second_hero': 'n/a',
                 'second_color': 'n/a', })
    random.shuffle(possible)
    return possible[:number]


def generate_data_for_kf_cnn(rounds):
    debug = False
    import datetime
    import time as timepackage
    kill_feed_hd5_path = os.path.join(cnn_kf_train_dir, 'dataset.hdf5')
    os.makedirs(cnn_kf_train_dir, exist_ok=True)
    if os.path.exists(kill_feed_hd5_path):
        print('skipping kf cnn data')
        return
    print('beginning kf cnn data')
    na_lab = 'n/a'
    # rounds = rounds[:3]
    # rounds = rounds[2:]
    set_files = {'first_hero': os.path.join(cnn_kf_train_dir, 'first_hero_set.txt'),
                 'first_color': os.path.join(cnn_kf_train_dir, 'first_color_set.txt'),
                 'ability': os.path.join(cnn_kf_train_dir, 'ability_set.txt'),
                 'second_hero': os.path.join(cnn_kf_train_dir, 'second_hero_set.txt'),
                 'second_color': os.path.join(cnn_kf_train_dir, 'second_color_set.txt'),
                 'spectator_mode': os.path.join(cnn_kf_train_dir, 'spectator_mode_set.txt')
                 }
    labels = [na_lab, 'assisting_hero'] + HERO_SET + ABILITY_SET + COLOR_SET
    print(labels)

    labs = ['first_hero', 'first_color', 'ability', 'second_hero', 'second_color']
    sets = {'first_hero': [na_lab] + HERO_SET,
            'first_color': [na_lab] + COLOR_SET,
            'ability': [na_lab] + ABILITY_SET,
            'second_hero': [na_lab] + HERO_SET,
            'second_color': [na_lab] + COLOR_SET}
    spectator_modes = [na_lab] + SPECTATOR_MODES
    time_step = 0.1
    params = BOX_PARAMETERS['O']['KILL_FEED_SLOT']
    # calc params
    num_frames = 0
    num_sequences = 0
    frames_per_seq = 100
    events = {}
    with open(os.path.join(cnn_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write(str(r['id']) + '\n')
            events[r['id']] = get_kf_events(r['id'])

            for beg, end in r['sequences']:
                expected_frame_count = int((end - beg) / time_step)
                num_frames += (int(expected_frame_count) + 1)
                num_sequences += (int(expected_frame_count / frames_per_seq) + 1) * 6

    print(num_sequences)
    indexes = random.sample(range(num_sequences), num_sequences)
    num_train = int(num_sequences * 0.8)
    num_val = num_sequences - num_train

    train_shape = (num_train, frames_per_seq, params['WIDTH'], params['HEIGHT'], 3)
    val_shape = (num_val, frames_per_seq, params['WIDTH'], params['HEIGHT'], 3)

    hdf5_file = h5py.File(kill_feed_hd5_path, mode='w')
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
        for k in sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count, frames_per_seq), np.uint8)
        hdf5_file.create_dataset("{}_label".format(pre), (count, frames_per_seq, params['WIDTH']), np.uint8)

    sequence_ind = 0
    for round_index, r in enumerate(rounds):
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
        spec_mode = r['spectator_mode'].lower()

        for beg, end in r['sequences']:
            print(beg, end)
            fvs = FileVideoStream(get_vod_path(r['stream_vod']), beg + r['begin'], end + r['begin'], time_step,
                                  real_begin=r['begin']).start()
            timepackage.sleep(1.0)

            print('begin main loop')
            begin_time = timepackage.time()
            images = {}
            data = {}
            label_datas = {}
            for slot in range(6):
                images[slot] = np.zeros((frames_per_seq, params['WIDTH'], params['HEIGHT'], 3),
                                        dtype=np.uint8)
                data[slot] = {}
                label_datas[slot] = []

                for k, s in sets.items():
                    data[slot][k] = np.zeros((frames_per_seq,), dtype=np.uint8)
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
                kf, e = construct_kf_at_time(events[r['id']], time_point)
                for slot in range(6):
                    x = params['X']
                    y = params['Y']
                    y += (params['HEIGHT'] + params['MARGIN']) * slot
                    box = frame[y: y + params['HEIGHT'],
                          x: x + params['WIDTH']]

                    images[slot][j, ...] = np.swapaxes(box, 1, 0)[None]
                    if slot < len(kf):
                        d = kf[slot]
                    else:
                        d = {x: na_lab for x in labs}
                    for k, s in sets.items():
                        d[k] = str(d[k]).lower()
                        data[slot][k][j] = sets[k].index(d[k])
                    label_data = []
                    if slot > len(kf) - 1 or kf[slot]['second_hero'] == na_lab:
                        label_data.append({'begin': 0, 'end': params['WIDTH'], 'label': na_lab})
                    else:
                        d = kf[slot]
                        print(d)
                        second_hero_left, second_hero_right = calculate_hero_boundaries(d['second_player'])

                        ability_left, ability_right = calculate_ability_boundaries(second_hero_left, d['ability'])
                        if d['first_player'] == 'n/a':
                            first_hero_pos = -1
                        else:
                            first_hero_left, first_hero_right = calculate_first_hero_boundaries(ability_left, len(
                                d['assisting_heroes']))
                            first_player_left, first_player_right = calculate_first_player_boundaries(first_hero_left,
                                                                                                      d['first_player'])
                            first_hero_left = params['WIDTH'] - first_hero_left
                            first_hero_right = params['WIDTH'] - first_hero_right
                            if d['assisting_heroes']:
                                assist_left, assist_right = calculate_assist_boundaries(ability_left, len(
                                    d['assisting_heroes']))
                                assist_left = params['WIDTH'] - assist_left
                                assist_right = params['WIDTH'] - assist_right
                                label_data.append(
                                    {'begin': assist_left, 'end': assist_right, 'label': 'assisting_hero'})

                                # cv2.imshow('frame_assist', box[:,assist_left: assist_right, :])
                        second_hero_left = params['WIDTH'] - second_hero_left
                        second_hero_right = params['WIDTH'] - second_hero_right
                        ability_left = params['WIDTH'] - ability_left
                        ability_right = params['WIDTH'] - ability_right

                        label_data.append({'begin': second_hero_right, 'end': params['WIDTH'],
                                           'label': d['second_color']})  # second name plate
                        label_data.append(
                            {'begin': second_hero_left, 'end': second_hero_right, 'label': d['second_hero']})
                        if d['headshot'] and not d['ability'].endswith('headshot'):
                            d['ability'] += ' headshot'
                        label_data.append({'begin': ability_left, 'end': ability_right, 'label': d['ability']})
                        label_data.append(
                            {'begin': ability_right, 'end': ability_right + 3, 'label': d['second_color']})

                        if d['first_player'] != 'n/a':
                            label_data.append(
                                {'begin': ability_left - 3, 'end': ability_left, 'label': d['first_color']})
                            first_player_left = params['WIDTH'] - first_player_left
                            if first_player_left < 0:
                                first_player_left = 0
                            else:
                                label_data.append({'begin': 0, 'end': first_player_left, 'label': na_lab})
                            first_player_right = params['WIDTH'] - first_player_right
                            if first_player_right > 0:
                                label_data.append(
                                    {'begin': first_player_left, 'end': first_hero_left, 'label': d['first_color']})
                            if first_hero_left < 0:
                                first_hero_left = 0
                            label_data.append(
                                {'begin': first_hero_left, 'end': first_hero_right, 'label': d['first_hero']})

                            # if first_player_left > 0:
                            #    cv2.imshow('frame_na_part', box[:,:first_player_left , :])
                            # cv2.imshow('frame_first_player', box[:,first_player_left: first_player_right, :])
                        label_data = sorted(label_data, key=lambda x: x['begin'])
                    label_datas[slot].append(label_data)

                j += 1
                if j == frames_per_seq:
                    for slot in range(6):
                        index = indexes[sequence_ind]
                        if index < num_train:
                            pre = 'train'
                        else:
                            pre = 'val'
                            index -= num_train
                        print(sequence_ind, num_sequences)
                        if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                            print('Train data: {}/{}'.format(sequence_ind, num_train))
                        elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                            print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                        hdf5_file["{}_img".format(pre)][index, ...] = images[slot][None]

                        images[slot] = np.zeros(
                            (frames_per_seq, params['WIDTH'], params['HEIGHT'], 3), dtype=np.uint8)
                        hdf5_file["{}_round".format(pre)][index] = r['id']
                        hdf5_file["{}_time_point".format(pre)][index] = time_point
                        hdf5_file["{}_spectator_mode".format(pre)][index] = spectator_modes.index(spec_mode)
                        for k, s in sets.items():
                            hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = data[slot][k][None]
                            data[slot][k] = np.zeros((frames_per_seq,), dtype=np.uint8)
                        for frame_ind in range(j):
                            for item in label_datas[slot][frame_ind]:
                                hdf5_file["{}_label".format(pre)][index, frame_ind,
                                item['begin']:item['end']] = labels.index(
                                    item['label'])
                        label_datas[slot] = []
                        sequence_ind += 1
                    j = 0

                    # if pre == 'train':
                    #    mean += box / num_train
                # if (r['id'], time_point) in error_set:
                #    error
            if j > 0:
                for player_ind in range(6):
                    index = indexes[sequence_ind]
                    if index < num_train:
                        pre = 'train'
                    else:
                        pre = 'val'
                        index -= num_train
                    if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                        print('Train data: {}/{}'.format(sequence_ind, num_train))
                    elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                        print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                    hdf5_file["{}_img".format(pre)][index, ...] = images[slot][None]
                    images[slot] = np.zeros(
                        (frames_per_seq, params['WIDTH'], params['HEIGHT'], 3), dtype=np.uint8)
                    hdf5_file["{}_round".format(pre)][index] = r['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    for k, s in sets.items():
                        hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = data[slot][k][None]
                        data[slot][k] = np.zeros((frames_per_seq,), dtype=np.uint8)
                    for frame_ind in range(j):
                        for item in label_datas[slot][frame_ind]:
                            hdf5_file["{}_label".format(pre)][index, frame_ind,
                            item['begin']:item['end']] = labels.index(
                                item['label'])
                    sequence_ind += 1

            print('main loop took', timepackage.time() - begin_time)

    hdf5_file.close()
    for k, v in set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in sets[k]:
                f.write('{}\n'.format(p))
    with open(os.path.join(cnn_kf_train_dir, 'labels.txt'), 'w', encoding='utf8') as f:
        for p in labels:
            f.write('{}\n'.format(p))
    with open(set_files['spectator_mode'], 'w', encoding='utf8') as f:
        for p in spectator_modes:
            f.write('{}\n'.format(p))

def generate_data_for_kf_slot_ctc_sequences(rounds):
    train_dir = os.path.join(training_data_directory, 'kf_slot_ctc_seq')
    debug = False
    import datetime
    import time as timepackage
    kill_feed_hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    os.makedirs(train_dir, exist_ok=True)
    if os.path.exists(kill_feed_hd5_path):
        print('skipping kf ctc seq data')
        return
    print('beginning kf ctc seq data')
    na_lab = 'n/a'
    frames_per_seq = 100
    # rounds = rounds[:3]
    labels = ['multiple_assists'] + [x + '_assist' for x in HERO_SET] + HERO_SET + ABILITY_SET + COLOR_SET
    time_step = 0.1
    params = BOX_PARAMETERS['O']['KILL_FEED_SLOT']
    # calc params
    num_frames = 0
    num_sequences = 0
    spectator_modes = [na_lab] + SPECTATOR_MODES
    events = {}
    ranges = {}
    with open(os.path.join(cnn_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write(str(r['id']) + '\n')
            events[r['id']] = get_kf_events(r['id'])
            ranges[r['id']] = get_event_ranges(events[r['id']], r['end'] - r['begin'])
            for rd in ranges[r['id']]:
                expected_duration = rd['end'] - rd['begin']
                expected_frame_count = expected_duration / time_step
                num_frames += (int(expected_frame_count) + 1)
                num_sequences += (int(expected_frame_count / frames_per_seq) + 1) * 6

    print(num_sequences)
    indexes = random.sample(range(num_sequences), num_sequences)
    num_train = int(num_sequences * 0.8)
    num_val = num_sequences - num_train

    train_shape = (num_train, frames_per_seq, params['WIDTH'], params['HEIGHT'], 3)
    val_shape = (num_val, frames_per_seq, params['WIDTH'], params['HEIGHT'], 3)

    hdf5_file = h5py.File(kill_feed_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        hdf5_file.create_dataset("{}_label_sequence".format(pre), (count, frames_per_seq, 6), np.uint8)
        hdf5_file.create_dataset("{}_label_sequence_length".format(pre), (count, frames_per_seq), np.uint8)
        hdf5_file.create_dataset("{}_spectator_mode".format(pre), (count,), np.uint8)

    sequence_ind = 0
    for round_index, r in enumerate(rounds):
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
        spec_mode = r['spectator_mode'].lower()
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['KILL_FEED_SLOT']

        for ra in ranges[r['id']]:
            begin_time = timepackage.time()
            fvs = FileVideoStream(get_vod_path(r['stream_vod']), ra['begin'] + r['begin'], ra['end'] + r['begin'],
                                  time_step,
                                  real_begin=r['begin']).start()
            timepackage.sleep(1.0)
            images = {}
            label_sequences = {}
            label_lengths = {}
            for slot in range(6):
                images[slot] = np.zeros((frames_per_seq, params['WIDTH'], params['HEIGHT'], 3),
                                        dtype=np.uint8)
                label_sequences[slot] = np.zeros((frames_per_seq, 6),
                                        dtype=np.uint8)
                label_lengths[slot] = np.zeros((frames_per_seq,),
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
                kf, e = construct_kf_at_time(events[r['id']], time_point)

                for slot in range(6):
                    x = params['X']
                    y = params['Y']
                    y += (params['HEIGHT'] + params['MARGIN']) * (slot)
                    box = frame[y: y + params['HEIGHT'],
                          x: x + params['WIDTH']]

                    images[slot][j, ...] = np.swapaxes(box, 1, 0)[None]
                    label_sequences[slot][j,:] = len(labels)
                    if slot > len(kf) - 1 or kf[slot]['second_hero'] == na_lab:
                        label_lengths[slot][j] = 1
                    else:
                        sequence = []
                        d = kf[slot]
                        if d['headshot'] and not d['ability'].endswith('headshot'):
                            d['ability'] += ' headshot'
                        second_hero_left, second_hero_right = calculate_hero_boundaries(d['second_player'])
                        ability_left, ability_right = calculate_ability_boundaries(second_hero_left, d['ability'])
                        if d['first_player'] == 'n/a':
                            pass
                        else:
                            first_hero_left, first_hero_right = calculate_first_hero_boundaries(ability_left, len(
                                d['assisting_heroes']))
                            first_player_left, first_player_right = calculate_first_player_boundaries(first_hero_left,
                                                                                                      d['first_player'])
                            first_player_left = params['WIDTH'] - first_player_left
                            first_player_right = params['WIDTH'] - first_player_right
                            if first_player_left > 0:
                                pass
                            if first_player_right > 0:
                                sequence.append(labels.index(d['first_color']))
                            sequence.append(labels.index(d['first_hero']))
                            if d['assisting_heroes']:
                                if len(d['assisting_heroes']) > 1:
                                    sequence.append(labels.index('multiple_assists'))
                                else:
                                    sequence.append(labels.index(d['assisting_heroes'][0].lower() + '_assist'))

                        sequence.append(labels.index(d['ability']))
                        sequence.append(labels.index(d['second_hero']))
                        sequence.append(labels.index(d['second_color']))
                        label_sequences[slot][j, 0:len(sequence)] = sequence
                        label_lengths[slot][j] = len(sequence)
                j += 1
                if j == frames_per_seq:
                    for slot in range(6):
                        index = indexes[sequence_ind]
                        if index < num_train:
                            pre = 'train'
                        else:
                            pre = 'val'
                            index -= num_train
                        print(sequence_ind, num_sequences)
                        if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                            print('Train data: {}/{}'.format(sequence_ind, num_train))
                        elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                            print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                        hdf5_file["{}_img".format(pre)][index, ...] = images[slot][None]
                        images[slot] = np.zeros(
                            (frames_per_seq, params['WIDTH'], params['HEIGHT'], 3), dtype=np.uint8)
                        hdf5_file["{}_round".format(pre)][index] = r['id']
                        hdf5_file["{}_time_point".format(pre)][index] = time_point

                        hdf5_file['{}_label_sequence'.format(pre)][index, ...] = label_sequences[slot][None]
                        hdf5_file['{}_label_sequence_length'.format(pre)][index, ...] = label_lengths[slot][None]
                        label_sequences[slot] = np.zeros((frames_per_seq, 6),
                                                dtype=np.uint8)
                        label_lengths[slot] = np.zeros((frames_per_seq,),
                                                dtype=np.uint8)

                        sequence_ind += 1
                    j = 0
            if j > 0:
                for slot in range(6):
                    index = indexes[sequence_ind]
                    if index < num_train:
                        pre = 'train'
                    else:
                        pre = 'val'
                        index -= num_train
                    print(sequence_ind, num_sequences)
                    if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                        print('Train data: {}/{}'.format(sequence_ind, num_train))
                    elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                        print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                    hdf5_file["{}_img".format(pre)][index, ...] = images[slot][None]
                    hdf5_file["{}_round".format(pre)][index] = r['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    for k in range(j, frames_per_seq):
                        label_lengths[slot][k] = 1
                        label_sequences[slot][k,:] = len(labels)
                    hdf5_file['{}_label_sequence'.format(pre)][index, ...] = label_sequences[slot][None]
                    hdf5_file['{}_label_sequence_length'.format(pre)][index, ...] = label_lengths[slot][None]


                    sequence_ind += 1

            print('main loop took', timepackage.time() - begin_time)

    hdf5_file.close()
    with open(os.path.join(train_dir, 'labels.txt'), 'w', encoding='utf8') as f:
        for p in labels:
            f.write('{}\n'.format(p))
    with open(os.path.join(train_dir, 'spectator_mode_set.txt'), 'w', encoding='utf8') as f:
        for p in spectator_modes:
            f.write('{}\n'.format(p))

def generate_data_for_kf_slot_ctc(rounds):
    train_dir = os.path.join(training_data_directory, 'kf_slot_ctc')
    debug = False
    import datetime
    import time as timepackage
    kill_feed_hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    os.makedirs(train_dir, exist_ok=True)
    if os.path.exists(kill_feed_hd5_path):
        print('skipping kf ctc data')
        return
    print('beginning kf ctc data')
    na_lab = 'n/a'
    #rounds = rounds[:3]
    labels = ['multiple_assists'] + [x + '_assist' for x in HERO_SET] + HERO_SET + ABILITY_SET + COLOR_SET
    time_step = 0.1
    params = BOX_PARAMETERS['O']['KILL_FEED_SLOT']
    # calc params
    num_frames = 0
    spectator_modes = [na_lab] + SPECTATOR_MODES
    events = {}
    ranges = {}
    with open(os.path.join(cnn_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write(str(r['id']) + '\n')
            events[r['id']] = get_kf_events(r['id'])
            ranges[r['id']] = get_event_ranges(events[r['id']], r['end'] - r['begin'])
            for rd in ranges[r['id']]:
                expected_duration = rd['end'] - rd['begin']
                expected_frame_count = expected_duration / time_step
                num_frames += (int(expected_frame_count) + 1) * 6

    print(num_frames)
    indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train

    train_shape = (num_train, params['WIDTH'], params['HEIGHT'], 3)
    val_shape = (num_val, params['WIDTH'], params['HEIGHT'], 3)

    hdf5_file = h5py.File(kill_feed_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        hdf5_file.create_dataset("{}_label_sequence".format(pre), (count, 12), np.uint8)
        hdf5_file.create_dataset("{}_label_sequence_length".format(pre), (count,), np.uint8)
        hdf5_file.create_dataset("{}_spectator_mode".format(pre), (count,), np.uint8)

    frame_ind = 0
    for round_index, r in enumerate(rounds):
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
        spec_mode = r['spectator_mode'].lower()
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['KILL_FEED_SLOT']
        begin_time = timepackage.time()
        fvs = FileVideoStreamRange(get_vod_path(r['stream_vod']), r['begin'], ranges[r['id']], time_step).start()
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break

            if frame_ind >= len(indexes):
                print('ignoring')
                frame_ind += 1
                continue
            time_point = round(time_point, 1)
            kf, e = construct_kf_at_time(events[r['id']], time_point)

            for slot in range(6):
                index = indexes[frame_ind]
                if index < num_train:
                    pre = 'train'
                else:
                    pre = 'val'
                    index -= num_train

                if frame_ind != 0 and (frame_ind) % 100 == 0 and frame_ind < num_train:
                    print('Train data: {}/{}'.format(frame_ind, num_train))
                elif frame_ind != 0 and frame_ind % 1000 == 0:
                    print('Validation data: {}/{}'.format(frame_ind - num_train, num_val))
                x = params['X']
                y = params['Y']
                y += (params['HEIGHT'] + params['MARGIN']) * (slot)
                box = frame[y: y + params['HEIGHT'],
                      x: x + params['WIDTH']]

                hdf5_file["{}_img".format(pre)][index, ...] = np.swapaxes(box, 1, 0)[None]
                hdf5_file["{}_label_sequence".format(pre)][index] = len(labels)
                if slot > len(kf) - 1 or kf[slot]['second_hero'] == na_lab:
                    hdf5_file["{}_label_sequence_length".format(pre)][index] = 1
                else:
                    sequence = []
                    d = kf[slot]
                    if d['headshot'] and not d['ability'].endswith('headshot'):
                        d['ability'] += ' headshot'
                    second_hero_left, second_hero_right = calculate_hero_boundaries(d['second_player'])
                    ability_left, ability_right = calculate_ability_boundaries(second_hero_left, d['ability'])
                    if d['first_player'] == 'n/a':
                        pass
                    else:
                        first_hero_left, first_hero_right = calculate_first_hero_boundaries(ability_left, len(
                            d['assisting_heroes']))
                        first_player_left, first_player_right = calculate_first_player_boundaries(first_hero_left,
                                                                                                  d['first_player'])
                        first_player_left = params['WIDTH'] - first_player_left
                        first_player_right = params['WIDTH'] - first_player_right
                        if first_player_left > 0:
                            pass
                        if first_player_right > 0:
                            sequence.append(labels.index(d['first_color']))
                        sequence.append(labels.index(d['first_hero']))
                        if d['assisting_heroes']:
                            for h in d['assisting_heroes']:
                                sequence.append(labels.index(h.lower() + '_assist'))

                    sequence.append(labels.index(d['ability']))

                    sequence.append(labels.index(d['second_hero']))
                    sequence.append(labels.index(d['second_color']))
                    hdf5_file["{}_label_sequence".format(pre)][index, 0:len(sequence)] = sequence
                    hdf5_file["{}_label_sequence_length".format(pre)][index] = len(sequence)
                hdf5_file["{}_round".format(pre)][index] = r['id']
                hdf5_file["{}_time_point".format(pre)][index] = time_point
                hdf5_file["{}_spectator_mode".format(pre)][index] = spectator_modes.index(spec_mode)
                frame_ind += 1

        print('main loop took', timepackage.time() - begin_time)
    while frame_ind < len(indexes):
        index = indexes[frame_ind]
        if index < num_train:
            pre = 'train'
        else:
            pre = 'val'
            index -= num_train
        hdf5_file["{}_label_sequence".format(pre)][index] = len(labels)
        hdf5_file["{}_label_sequence_length".format(pre)][index] = 1
        frame_ind += 1
    hdf5_file.close()
    with open(os.path.join(train_dir, 'labels.txt'), 'w', encoding='utf8') as f:
        for p in labels:
            f.write('{}\n'.format(p))
    with open(os.path.join(train_dir, 'spectator_mode_set.txt'), 'w', encoding='utf8') as f:
        for p in spectator_modes:
            f.write('{}\n'.format(p))


def generate_data_for_kf_slot_gru(rounds):
    train_dir = os.path.join(training_data_directory, 'kf_slot_gru')
    debug = False
    import datetime
    import time as timepackage
    kill_feed_hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    os.makedirs(train_dir, exist_ok=True)
    if os.path.exists(kill_feed_hd5_path):
        print('skipping kf gru data')
        return
    print('beginning kf gru data')
    na_lab = 'n/a'
    # rounds = rounds[:3]
    # rounds = rounds[2:]

    set_files = {'first_hero': os.path.join(train_dir, 'first_hero_set.txt'),
                 'first_color': os.path.join(train_dir, 'first_color_set.txt'),
                 'ability': os.path.join(train_dir, 'ability_set.txt'),
                 'second_hero': os.path.join(train_dir, 'second_hero_set.txt'),
                 'second_color': os.path.join(train_dir, 'second_color_set.txt'),
                 'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt')
                 }
    labels = [na_lab, 'multiple_assists'] + [x + '_assist' for x in HERO_SET] + HERO_SET + ABILITY_SET + COLOR_SET
    print(labels)

    labs = ['first_hero', 'first_color', 'ability', 'headshot', 'second_hero', 'second_color']
    sets = {'first_hero': [na_lab] + HERO_SET,
            'first_color': [na_lab] + COLOR_SET,
            'ability': [na_lab] + ABILITY_SET,
            'second_hero': [na_lab] + HERO_SET,
            'second_color': [na_lab] + COLOR_SET,
            'spectator_mode': [na_lab] + SPECTATOR_MODES}

    time_step = 0.1
    params = BOX_PARAMETERS['O']['KILL_FEED_SLOT']
    # calc params
    num_frames = 0
    frames_per_seq = 100
    events = {}
    ranges = {}
    num_sequences = 0
    with open(os.path.join(cnn_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write(str(r['id']) + '\n')
            events[r['id']] = get_kf_events(r['id'])
            ranges[r['id']] = get_event_ranges(events[r['id']], r['end'] - r['begin'])
            for rd in ranges[r['id']]:
                expected_duration = rd['end'] - rd['begin']
                expected_frame_count = expected_duration / time_step
                num_frames += (int(expected_frame_count) + 1) * 6
                num_sequences += (int(expected_frame_count / frames_per_seq) + 1) * 6

    print(num_sequences)
    indexes = random.sample(range(num_sequences), num_sequences)
    num_train = int(num_sequences * 0.8)
    num_val = num_sequences - num_train

    train_shape = (num_train, frames_per_seq, params['WIDTH'], params['HEIGHT'], 3)
    val_shape = (num_val, frames_per_seq, params['WIDTH'], params['HEIGHT'], 3)

    hdf5_file = h5py.File(kill_feed_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        hdf5_file.create_dataset("{}_label".format(pre), (count, frames_per_seq, params['WIDTH']), np.uint8)
        for k in sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count, frames_per_seq), np.uint8)

    sequence_ind = 0
    for round_index, r in enumerate(rounds):
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
        spec_mode = r['spectator_mode'].lower()
        params = BOX_PARAMETERS[r['game']['match']['film_format']]['KILL_FEED_SLOT']

        for ra in ranges[r['id']]:
            begin_time = timepackage.time()
            fvs = FileVideoStream(get_vod_path(r['stream_vod']), ra['begin'] + r['begin'], ra['end'] + r['begin'],
                                  time_step,
                                  real_begin=r['begin']).start()
            timepackage.sleep(1.0)
            images = {}
            data = {}
            for slot in range(6):
                images[slot] = np.zeros((frames_per_seq, params['WIDTH'], params['HEIGHT'], 3),
                                        dtype=np.uint8)
                data[slot] = {}
                for k, s in sets.items():
                    data[slot][k] = np.zeros((frames_per_seq,), dtype=np.uint8)
                data[slot]['labels'] = np.zeros((frames_per_seq, params['WIDTH']), dtype=np.uint8)
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
                kf, e = construct_kf_at_time(events[r['id']], time_point)

                for slot in range(6):
                    x = params['X']
                    y = params['Y']
                    y += (params['HEIGHT'] + params['MARGIN']) * (slot)
                    box = frame[y: y + params['HEIGHT'],
                          x: x + params['WIDTH']]

                    images[slot][j, ...] = np.swapaxes(box, 1, 0)[None]
                    label_data = []
                    if slot > len(kf) - 1 or kf[slot]['second_hero'] == na_lab:
                        label_data.append({'begin': 0, 'end': params['WIDTH'], 'label': na_lab})
                        for k in sets.keys():
                            if k == 'spectator_mode':
                                data[slot][k][j] = sets[k].index(spec_mode)
                            else:
                                data[slot][k][j] = sets[k].index(na_lab)
                    else:
                        d = kf[slot]
                        d['spectator_mode'] = spec_mode
                        second_hero_left, second_hero_right = calculate_hero_boundaries(d['second_player'])

                        ability_left, ability_right = calculate_ability_boundaries(second_hero_left, d['ability'])
                        if d['first_player'] == 'n/a':
                            first_hero_pos = -1
                        else:
                            first_hero_left, first_hero_right = calculate_first_hero_boundaries(ability_left, len(
                                d['assisting_heroes']))
                            first_player_left, first_player_right = calculate_first_player_boundaries(first_hero_left,
                                                                                                      d['first_player'])
                            first_hero_left = params['WIDTH'] - first_hero_left
                            first_hero_right = params['WIDTH'] - first_hero_right
                            if d['assisting_heroes']:
                                if len(d['assisting_heroes']) > 1:
                                    assist_lab = 'multiple_assists'
                                else:
                                    assist_lab = d['assisting_heroes'][0].lower() + '_assist'
                                assist_left, assist_right = calculate_assist_boundaries(ability_left, len(
                                    d['assisting_heroes']))
                                assist_left = params['WIDTH'] - assist_left
                                assist_right = params['WIDTH'] - assist_right
                                label_data.append({'begin': assist_left, 'end': assist_right, 'label': assist_lab})

                                # cv2.imshow('frame_assist', box[:,assist_left: assist_right, :])
                        second_hero_left = params['WIDTH'] - second_hero_left
                        second_hero_right = params['WIDTH'] - second_hero_right
                        ability_left = params['WIDTH'] - ability_left
                        ability_right = params['WIDTH'] - ability_right

                        label_data.append({'begin': second_hero_right, 'end': params['WIDTH'],
                                           'label': d['second_color']})  # second name plate
                        label_data.append(
                            {'begin': second_hero_left, 'end': second_hero_right, 'label': d['second_hero']})
                        if d['headshot'] and not d['ability'].endswith('headshot'):
                            d['ability'] += ' headshot'
                        label_data.append({'begin': ability_left, 'end': ability_right, 'label': d['ability']})
                        label_data.append(
                            {'begin': ability_right, 'end': ability_right + 3, 'label': d['second_color']})

                        if d['first_player'] != 'n/a':
                            label_data.append(
                                {'begin': ability_left - 3, 'end': ability_left, 'label': d['first_color']})
                            first_player_left = params['WIDTH'] - first_player_left
                            if first_player_left < 0:
                                first_player_left = 0
                            else:
                                label_data.append({'begin': 0, 'end': first_player_left, 'label': na_lab})
                            first_player_right = params['WIDTH'] - first_player_right
                            if first_player_right > 0:
                                label_data.append(
                                    {'begin': first_player_left, 'end': first_hero_left, 'label': d['first_color']})
                            if first_hero_left < 0:
                                first_hero_left = 0
                            label_data.append(
                                {'begin': first_hero_left, 'end': first_hero_right, 'label': d['first_hero']})

                            # if first_player_left > 0:
                            #    cv2.imshow('frame_na_part', box[:,:first_player_left , :])
                            # cv2.imshow('frame_first_player', box[:,first_player_left: first_player_right, :])
                        label_data = sorted(label_data, key=lambda x: x['begin'])
                        for item in label_data:
                            data[slot]['labels'][j, item['begin']:item['end']] = labels.index(item['label'])

                        for k in sets.keys():
                            data[slot][k][j] = sets[k].index(d[k])

                        # cv2.imshow('frame', box)
                        # cv2.imshow('frame_first_hero', box[:,first_hero_left: first_hero_right, :])
                        # cv2.imshow('frame_second_hero', box[:,second_hero_left: second_hero_right, :])
                        # cv2.imshow('frame_second_player', box[:,second_hero_right: , :])
                        # cv2.imshow('frame_ability'   , box[:,ability_left: ability_right, :])
                        # cv2.waitKey(0)

                    # if d['first_hero'] != 'n/a':
                    #    cv2.imshow('frame_{}'.format(slot), box)
                    #    print(first_hero_left, first_hero_right)
                    #    print(second_hero_left, second_hero_right)
                    #    print(ability_left, ability_right)
                j += 1
                if j == frames_per_seq:
                    for slot in range(6):
                        index = indexes[sequence_ind]
                        if index < num_train:
                            pre = 'train'
                        else:
                            pre = 'val'
                            index -= num_train
                        print(sequence_ind, num_sequences)
                        if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                            print('Train data: {}/{}'.format(sequence_ind, num_train))
                        elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                            print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                        hdf5_file["{}_img".format(pre)][index, ...] = images[slot][None]
                        images[slot] = np.zeros(
                            (frames_per_seq, params['WIDTH'], params['HEIGHT'], 3), dtype=np.uint8)
                        hdf5_file["{}_round".format(pre)][index] = r['id']
                        hdf5_file["{}_time_point".format(pre)][index] = time_point
                        for k, s in sets.items():
                            hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = data[slot][k][None]
                            data[slot][k] = np.zeros((frames_per_seq,), dtype=np.uint8)
                        hdf5_file['{}_label'.format(pre)][index, ...] = data[slot]['labels'][None]
                        data[slot]['labels'] = np.zeros((frames_per_seq, params['WIDTH']), dtype=np.uint8)

                        sequence_ind += 1
                    j = 0
            if j > 0:
                for slot in range(6):
                    index = indexes[sequence_ind]
                    if index < num_train:
                        pre = 'train'
                    else:
                        pre = 'val'
                        index -= num_train
                    print(sequence_ind, num_sequences)
                    if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                        print('Train data: {}/{}'.format(sequence_ind, num_train))
                    elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                        print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                    hdf5_file["{}_img".format(pre)][index, ...] = images[slot][None]
                    hdf5_file["{}_round".format(pre)][index] = r['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    for k, s in sets.items():
                        hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = data[slot][k][None]

                    hdf5_file['{}_label'.format(pre)][index, ...] = data[slot]['labels'][None]

                    sequence_ind += 1

            print('main loop took', timepackage.time() - begin_time)

    hdf5_file.close()
    with open(os.path.join(train_dir, 'labels.txt'), 'w', encoding='utf8') as f:
        for p in labels:
            f.write('{}\n'.format(p))
    for k, v in sets.items():
        with open(set_files[k], 'w', encoding='utf8') as f:
            for p in v:
                f.write('{}\n'.format(p))


def generate_data_for_kf_cnn_slot(rounds):
    train_dir = os.path.join(training_data_directory, 'kf_cnn_slot')
    debug = False
    import datetime
    import time as timepackage
    kill_feed_hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    os.makedirs(train_dir, exist_ok=True)
    if os.path.exists(kill_feed_hd5_path):
        print('skipping kf cnn data')
        return
    print('beginning kf cnn data')
    na_lab = 'n/a'
    # rounds = rounds[:3]
    # rounds = rounds[2:]

    set_files = {'first_hero': os.path.join(train_dir, 'first_hero_set.txt'),
                 'first_color': os.path.join(train_dir, 'first_color_set.txt'),
                 'ability': os.path.join(train_dir, 'ability_set.txt'),
                 'second_hero': os.path.join(train_dir, 'second_hero_set.txt'),
                 'second_color': os.path.join(train_dir, 'second_color_set.txt'),
                 'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt')
                 }
    labels = [na_lab, 'multiple_assists'] + [x + '_assist' for x in HERO_SET] + HERO_SET + ABILITY_SET + COLOR_SET
    print(labels)

    labs = ['first_hero', 'first_color', 'ability', 'headshot', 'second_hero', 'second_color']
    sets = {'first_hero': [na_lab] + HERO_SET,
            'first_color': [na_lab] + COLOR_SET,
            'ability': [na_lab] + ABILITY_SET,
            'second_hero': [na_lab] + HERO_SET,
            'second_color': [na_lab] + COLOR_SET,
            # 'spectator_mode': [na_lab] + SPECTATOR_MODES
            }

    spectator_modes = [na_lab] + SPECTATOR_MODES
    time_step = 0.1
    params = BOX_PARAMETERS['O']['KILL_FEED_SLOT']
    # calc params
    num_frames = 0

    events = {}
    ranges = {}
    with open(os.path.join(cnn_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write(str(r['id']) + '\n')
            events[r['id']] = get_kf_events(r['id'])
            ranges[r['id']] = get_event_ranges(events[r['id']], r['end'] - r['begin'])
            for rd in ranges[r['id']]:
                expected_duration = rd['end'] - rd['begin']
                expected_frame_count = expected_duration / time_step
                num_frames += (int(expected_frame_count) + 1) * 6

    print(num_frames)
    indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train

    train_shape = (num_train, params['WIDTH'], params['HEIGHT'], 3)
    val_shape = (num_val, params['WIDTH'], params['HEIGHT'], 3)

    hdf5_file = h5py.File(kill_feed_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        hdf5_file.create_dataset("{}_label".format(pre), (count, int(params['WIDTH'] / 4)), np.uint8)
        hdf5_file.create_dataset("{}_spectator_mode".format(pre), (count,), np.uint8)
        for k in sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count,), np.uint8)

    frame_ind = 0
    for round_index, r in enumerate(rounds):
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
        spec_mode = r['spectator_mode'].lower()
        params = BOX_PARAMETERS[r['game']['match']['film_format']]['KILL_FEED_SLOT']
        begin_time = timepackage.time()
        fvs = FileVideoStreamRange(get_vod_path(r['stream_vod']), r['begin'], ranges[r['id']], time_step).start()
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break

            if frame_ind >= len(indexes):
                print('ignoring')
                frame_ind += 1
                continue
            time_point = round(time_point, 1)
            kf, e = construct_kf_at_time(events[r['id']], time_point)

            for slot in range(6):
                index = indexes[frame_ind]
                if index < num_train:
                    pre = 'train'
                else:
                    pre = 'val'
                    index -= num_train

                if frame_ind != 0 and (frame_ind) % 100 == 0 and frame_ind < num_train:
                    print('Train data: {}/{}'.format(frame_ind, num_train))
                elif frame_ind != 0 and frame_ind % 1000 == 0:
                    print('Validation data: {}/{}'.format(frame_ind - num_train, num_val))
                x = params['X']
                y = params['Y']
                y += (params['HEIGHT'] + params['MARGIN']) * (slot)
                box = frame[y: y + params['HEIGHT'],
                      x: x + params['WIDTH']]

                hdf5_file["{}_img".format(pre)][index, ...] = np.swapaxes(box, 1, 0)[None]
                hdf5_file['{}_spectator_mode'.format(pre)][index] = spectator_modes.index(spec_mode)
                hdf5_file['{}_round'.format(pre)][index] = r['id']
                hdf5_file['{}_time_point'.format(pre)][index] = time_point
                label_data = []
                if slot > len(kf) - 1 or kf[slot]['second_hero'] == na_lab:
                    label_data.append({'begin': 0, 'end': int(round(params['WIDTH'] / 4)), 'label': na_lab})
                else:
                    d = kf[slot]
                    second_hero_left, second_hero_right = calculate_hero_boundaries(d['second_player'])

                    ability_left, ability_right = calculate_ability_boundaries(second_hero_left, d['ability'])
                    if d['first_player'] == 'n/a':
                        first_hero_pos = -1
                    else:
                        first_hero_left, first_hero_right = calculate_first_hero_boundaries(ability_left, len(
                            d['assisting_heroes']))
                        first_player_left, first_player_right = calculate_first_player_boundaries(first_hero_left,
                                                                                                  d['first_player'])
                        first_hero_left = params['WIDTH'] - first_hero_left
                        first_hero_right = params['WIDTH'] - first_hero_right
                        if d['assisting_heroes']:
                            if len(d['assisting_heroes']) > 1:
                                assist_lab = 'multiple_assists'
                            else:
                                assist_lab = d['assisting_heroes'][0].lower() + '_assist'
                            assist_left, assist_right = calculate_assist_boundaries(ability_left, len(
                                d['assisting_heroes']))
                            assist_left = params['WIDTH'] - assist_left
                            assist_right = params['WIDTH'] - assist_right
                            label_data.append(
                                {'begin': int(round(assist_left / 4)), 'end': int(round(assist_right / 4)),
                                 'label': assist_lab})

                            # cv2.imshow('frame_assist', box[:,assist_left: assist_right, :])
                    second_hero_left = params['WIDTH'] - second_hero_left
                    second_hero_right = params['WIDTH'] - second_hero_right
                    ability_left = params['WIDTH'] - ability_left
                    ability_right = params['WIDTH'] - ability_right

                    label_data.append(
                        {'begin': int(round(second_hero_right / 4)), 'end': int(round(params['WIDTH'] / 4)),
                         'label': d['second_color']})  # second name plate
                    label_data.append(
                        {'begin': int(round(second_hero_left / 4)), 'end': int(round(second_hero_right / 4)),
                         'label': d['second_hero']})
                    if d['headshot'] and not d['ability'].endswith('headshot'):
                        d['ability'] += ' headshot'
                    label_data.append({'begin': int(round(ability_left / 4)), 'end': int(round(ability_right / 4)),
                                       'label': d['ability']})
                    label_data.append(
                        {'begin': int(round(ability_right / 4)), 'end': int(round((ability_right + 3) / 4)),
                         'label': d['second_color']})

                    if d['first_player'] != 'n/a':
                        label_data.append(
                            {'begin': int(round((ability_left - 3) / 4)), 'end': int(round(ability_left / 4)),
                             'label': d['first_color']})
                        first_player_left = params['WIDTH'] - first_player_left
                        if first_player_left < 0:
                            first_player_left = 0
                        else:
                            label_data.append({'begin': 0, 'end': int(round(first_player_left / 4)), 'label': na_lab})
                        first_player_right = params['WIDTH'] - first_player_right
                        if first_player_right > 0:
                            label_data.append(
                                {'begin': int(round(first_player_left / 4)), 'end': int(round(first_hero_left / 4)),
                                 'label': d['first_color']})
                        if first_hero_left < 0:
                            first_hero_left = 0
                        label_data.append(
                            {'begin': int(round(first_hero_left / 4)), 'end': int(round(first_hero_right / 4)),
                             'label': d['first_hero']})

                        # if first_player_left > 0:
                        #    cv2.imshow('frame_na_part', box[:,:first_player_left , :])
                        # cv2.imshow('frame_first_player', box[:,first_player_left: first_player_right, :])
                    label_data = sorted(label_data, key=lambda x: x['begin'])
                    for item in label_data:
                        try:
                            hdf5_file["{}_label".format(pre)][index, item['begin']:item['end']] = labels.index(
                                item['label'])
                        except ValueError:
                            cv2.imshow('frame', box)
                            print(label_data)
                            cv2.waitKey(0)
                            raise

                    for k in sets.keys():
                        hdf5_file["{}_{}_label".format(pre, k)][index] = sets[k].index(d[k])

                    # cv2.imshow('frame', box)
                    # cv2.imshow('frame_first_hero', box[:,first_hero_left: first_hero_right, :])
                    # cv2.imshow('frame_second_hero', box[:,second_hero_left: second_hero_right, :])
                    # cv2.imshow('frame_second_player', box[:,second_hero_right: , :])
                    # cv2.imshow('frame_ability'   , box[:,ability_left: ability_right, :])
                    # cv2.waitKey(0)

                # if d['first_hero'] != 'n/a':
                #    cv2.imshow('frame_{}'.format(slot), box)
                #    print(first_hero_left, first_hero_right)
                #    print(second_hero_left, second_hero_right)
                #    print(ability_left, ability_right)
                frame_ind += 1

        print('main loop took', timepackage.time() - begin_time)

    hdf5_file.close()
    with open(os.path.join(train_dir, 'labels.txt'), 'w', encoding='utf8') as f:
        for p in labels:
            f.write('{}\n'.format(p))
    for k, v in sets.items():
        with open(set_files[k], 'w', encoding='utf8') as f:
            for p in v:
                f.write('{}\n'.format(p))


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
        params = BOX_PARAMETERS[r['game']['match']['film_format']]['REPLAY']
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
        params = BOX_PARAMETERS[r['game']['match']['film_format']]['PAUSE']
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
    hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    os.makedirs(train_dir, exist_ok=True)
    if os.path.exists(hd5_path):
        print('skipping game cnn data')
        return
    print('beginning game cnn data')
    error_set = []
    num_frames = 0
    num_sequences = 0
    frames_per_seq = 100
    seqs = {}
    for v in vods:
        seqs[v['id']] = []
        cap = cv2.VideoCapture(get_vod_path(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur = num_frames / fps
        cap.release()
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
    print(num_sequences)
    indexes = random.sample(range(num_sequences), num_sequences)
    num_train = int(num_sequences * 0.8)
    num_val = num_sequences - num_train

    params = BOX_PARAMETERS['O']['MID']
    train_shape = (num_train, frames_per_seq, int(params['HEIGHT'] * 0.5), int(params['WIDTH'] * 0.5), 3)
    val_shape = (num_val, frames_per_seq, int(params['HEIGHT'] * 0.5), int(params['WIDTH'] * 0.5), 3)

    labels = ['not_in_game', 'game']

    hdf5_file = h5py.File(hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_vod".format(pre), (count,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        hdf5_file.create_dataset("{}_label".format(pre), (count, frames_per_seq), np.uint8)
        hdf5_file.create_dataset("{}_prev_label".format(pre), (count,), np.uint8)

    print(num_frames, num_train, num_val)
    sequence_ind = 0
    for i, v in enumerate(vods):
        print(i, len(vods))
        for seq in seqs[v['id']]:
            beg, end = seq
            print(beg, end)
            fvs = FileVideoStream(get_vod_path(v), beg, end, time_step, real_begin=0).start()
            timepackage.sleep(1.0)
            prev_label = labels[0]
            begin_time = timepackage.time()
            data = np.zeros((frames_per_seq,), dtype=np.uint8)
            images = np.zeros((frames_per_seq, int(params['HEIGHT'] * 0.5), int(params['WIDTH'] * 0.5), 3),
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
                print(time_point)
                lab = labels[0]
                for s in v['sequences']:
                    if s[0] - 2 <= time_point <= s[1] + 2:
                        lab = labels[1]

                data[j] = labels.index(lab)
                x = params['X']
                box = frame[params['Y']: params['Y'] + params['HEIGHT'],
                      x: x + params['WIDTH']]
                box = cv2.resize(box, (0, 0), fx=0.5, fy=0.5)

                images[j, ...] = box[None]
                j += 1
                if j == frames_per_seq:
                    index = indexes[sequence_ind]
                    if index < num_train:
                        pre = 'train'
                    else:
                        pre = 'val'
                        index -= num_train
                    print(sequence_ind, num_sequences)
                    if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                        print('Train data: {}/{}'.format(sequence_ind, num_train))
                    elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                        print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                    hdf5_file["{}_img".format(pre)][index, ...] = images[None]

                    images = np.zeros(
                        (frames_per_seq, int(params['HEIGHT'] * 0.5), int(params['WIDTH'] * 0.5), 3), dtype=np.uint8)
                    hdf5_file["{}_vod".format(pre)][index] = v['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    hdf5_file['{}_label'.format(pre)][index, ...] = data[None]
                    hdf5_file['{}_prev_label'.format(pre)][index] = labels.index(prev_label)
                    prev_label = labels[data[-1]]
                    data = np.zeros((frames_per_seq,), dtype=np.uint8)

                    sequence_ind += 1
                    j = 0
            if j > 0:
                index = indexes[sequence_ind]
                if index < num_train:
                    pre = 'train'
                else:
                    pre = 'val'
                    index -= num_train
                print(sequence_ind, num_sequences)
                if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                    print('Train data: {}/{}'.format(sequence_ind, num_train))
                elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                    print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                hdf5_file["{}_img".format(pre)][index, ...] = images[None]

                hdf5_file["{}_vod".format(pre)][index] = v['id']
                hdf5_file["{}_time_point".format(pre)][index] = time_point
                hdf5_file['{}_label'.format(pre)][index, ...] = data[None]
                hdf5_file['{}_prev_label'.format(pre)][index] = labels.index(prev_label)

                sequence_ind += 1
            print('main loop took', timepackage.time() - begin_time)

    hdf5_file.close()
    with open(os.path.join(train_dir, 'labels.txt'), 'w', encoding='utf8') as f:
        for p in labels:
            f.write('{}\n'.format(p))


def generate_data_for_round_cnn(rounds):
    import time as timepackage
    debug = False
    train_dir = os.path.join(training_data_directory, 'round_cnn')
    status_hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    set_files = {
        'left_color': os.path.join(train_dir, 'color_set.txt'),
        'right_color': os.path.join(train_dir, 'color_set.txt'),
        'map': os.path.join(train_dir, 'map_set.txt'),
        'map_mode': os.path.join(train_dir, 'map_mode_set.txt'),
        'round_number': os.path.join(train_dir, 'round_number_set.txt'),
        'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
    }
    na_lab = 'n/a'
    sets = {
        'left_color': [na_lab] + COLOR_SET,
        'right_color': [na_lab] + COLOR_SET,
        'map': [na_lab] + MAP_SET,
        'map_mode': [na_lab] + MAP_MODE_SET,
        'round_number': range(1, 7),
        'spectator_mode': [na_lab] + SPECTATOR_MODES
    }
    os.makedirs(train_dir, exist_ok=True)
    if os.path.exists(status_hd5_path):
        print('skipping rounds cnn data')
        return
    print('beginning rounds cnn data')
    error_set = []
    # calc params
    num_frames = 0
    num_sequences = len(rounds)
    frames_per_seq = 200
    print(num_sequences)
    indexes = random.sample(range(num_sequences), num_sequences)
    num_train = int(num_sequences * 0.8)
    num_val = num_sequences - num_train

    params = BOX_PARAMETERS['O']['MID']
    resize_factor = 0.5
    train_shape = (
        num_train, frames_per_seq, int(params['HEIGHT'] * resize_factor), int(params['WIDTH'] * resize_factor), 3)
    val_shape = (
        num_val, frames_per_seq, int(params['HEIGHT'] * resize_factor), int(params['WIDTH'] * resize_factor), 3)

    hdf5_file = h5py.File(status_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32)
        for k in sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count,), np.uint8)

    print(num_frames, num_train, num_val)
    sequence_ind = 0
    for i, r in enumerate(rounds):
        print(i, len(rounds))
        if debug:
            # if r['game']['match']['film_format'] != '2':
            #    continue
            if error_set and r['id'] not in [x[0] for x in error_set]:
                continue
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'],
              r['game']['match']['film_format'])
        params = BOX_PARAMETERS[r['game']['match']['film_format']]['MID']
        spec_mode = r['spectator_mode'].lower()

        left_color = r['game']['left_team']['color'].lower()

        right_color = r['game']['right_team']['color'].lower()
        map = r['game']['map']['name'].lower()
        map_mode = r['game']['map']['mode'].lower()
        dur = r['end'] - r['begin']
        time_step = dur / (frames_per_seq)

        fvs = FileVideoStream(get_vod_path(r['stream_vod']), r['begin'], r['end'], time_step,
                              real_begin=0).start()
        data = {'left_color': left_color, 'right_color': right_color, 'map': map, 'spectator_mode': spec_mode,
                'map_mode': map_mode, 'round_number': r['round_number']}
        images = np.zeros(
            (frames_per_seq, int(params['HEIGHT'] * resize_factor), int(params['WIDTH'] * resize_factor), 3),
            dtype=np.uint8)
        j = 0
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            time_point = round(time_point, 1)
            print(j)
            x = params['X']
            box = frame[params['Y']: params['Y'] + params['HEIGHT'],
                  x: x + params['WIDTH']]
            box = cv2.resize(box, (0, 0), fx=resize_factor, fy=resize_factor)
            # cv2.imshow('frame', box)
            # print(data)
            # cv2.waitKey(0)
            if j > frames_per_seq - 1:
                continue
            images[j, ...] = box[None]
            j += 1
        index = indexes[sequence_ind]
        if index < num_train:
            pre = 'train'
        else:
            pre = 'val'
            index -= num_train
        print(sequence_ind, num_sequences)
        if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
            print('Train data: {}/{}'.format(sequence_ind, num_train))
        elif sequence_ind != 0 and sequence_ind % 1000 == 0:
            print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
        hdf5_file["{}_img".format(pre)][index, ...] = images[None]

        hdf5_file["{}_round".format(pre)][index] = r['id']
        for k, s in sets.items():
            d = s.index(data[k])
            hdf5_file['{}_{}_label'.format(pre, k)][index] = d

        sequence_ind += 1

    hdf5_file.close()
    for k, v in set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in sets[k]:
                f.write('{}\n'.format(p))


def generate_data_for_cnn(rounds, vods, rounds_plus):
    generate_data_for_replay_cnn(rounds)
    generate_data_for_pause_cnn(rounds)
    generate_data_for_kf_cnn_slot(rounds)
    generate_data_for_kf_slot_gru(rounds)
    generate_data_for_player_cnn(rounds)
    generate_data_for_kf_cnn(rounds)
    generate_data_for_kf_slot_ctc(rounds)
    generate_data_for_mid_cnn(rounds)
    generate_data_for_game_cnn(vods)
    generate_data_for_round_cnn(rounds_plus)
    generate_data_for_kf_slot_ctc_sequences(rounds)


def save_round_info(rounds):
    with open(os.path.join(training_data_directory, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write('{} {} {}\n'.format(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number']))


if __name__ == '__main__':
    rounds_plus = get_train_rounds_plus()
    rounds = get_train_rounds()
    for r in rounds_plus:
        print(r['sequences'])
        local_path = get_vod_path(r['stream_vod'])
        if not os.path.exists(local_path):
            if get_local_path(r) is not None:
                shutil.move(get_local_path(r), local_path)
            else:
                print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
                get_local_file(r)
    save_round_info(rounds)

    vods = get_train_vods()
    generate_data_for_cnn(rounds, vods, rounds_plus)
