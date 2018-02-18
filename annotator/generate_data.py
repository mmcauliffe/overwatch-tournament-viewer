import requests
import os
import numpy as np
import h5py
import random
import cv2

from annotator.utils import get_local_file, HERO_SET, ABILITY_SET, BOX_PARAMETERS, get_player_states, get_local_path, \
    get_kf_events, get_round_states, get_train_rounds, FileVideoStream, look_up_player_state, look_up_round_state, \
    load_set, Empty, calculate_ability_boundaries, calculate_first_hero_boundaries, calculate_hero_boundaries, FileVideoStreamRange, get_event_ranges, COLOR_SET,calculate_first_player_boundaries, calculate_assist_boundaries

training_data_directory = r'E:\Data\Overwatch\training_data'
model_directory = r'E:\Data\Overwatch\models'
cnn_status_model_dir = os.path.join(model_directory, 'player_status_cnn')
cnn_mid_model_dir = os.path.join(model_directory, 'mid_cnn')
cnn_kf_model_dir = os.path.join(model_directory, 'kf_cnn')
cnn_status_train_dir = os.path.join(training_data_directory, 'player_status_cnn')
lstm_status_train_dir = os.path.join(training_data_directory, 'player_status_lstm')
cnn_mid_train_dir = os.path.join(training_data_directory, 'mid_cnn')
lstm_mid_train_dir = os.path.join(training_data_directory, 'mid_lstm')
cnn_kf_train_dir = os.path.join(training_data_directory, 'kf_cnn')
lstm_kf_train_dir = os.path.join(training_data_directory, 'kf_lstm')
model_directory = r'E:\Data\Overwatch\models'


def generate_data_for_player_lstm(rounds):
    time_step = 0.1
    frames_per_seq = 100
    embedding_size = 50
    import keras
    final_output_weights = os.path.join(cnn_status_model_dir, 'player_weights.h5')
    final_output_json = os.path.join(cnn_status_model_dir, 'player_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    embedding_model = keras.models.Model(inputs=model.input,
                                         outputs=model.get_layer('representation').output)
    import time as timepackage
    os.makedirs(lstm_status_train_dir, exist_ok=True)
    status_hd5_path = os.path.join(lstm_status_train_dir, 'dataset.hdf5')
    if os.path.exists(status_hd5_path):
        print('skipping player lstm data')
        return
    print('beginning player lstm data')
    na_lab = 'n/a'
    set_files = {'player': os.path.join(cnn_status_train_dir, 'player_set.txt'),
                 'hero': os.path.join(cnn_status_train_dir, 'hero_set.txt'),
                 'color': os.path.join(cnn_status_train_dir, 'color_set.txt'),
                 'alive': os.path.join(cnn_status_train_dir, 'alive_set.txt'),
                 'ult': os.path.join(cnn_status_train_dir, 'ult_set.txt'),
                 }
    sets = {}
    for k, v in set_files.items():
        if k == 'hero':
            sets[k] = [na_lab] + HERO_SET
        else:
            sets[k] = load_set(v)
            if na_lab not in sets[k]:
                sets[k].insert(0, na_lab)

    left_params = BOX_PARAMETERS['REGULAR']['LEFT']
    right_params = BOX_PARAMETERS['REGULAR']['RIGHT']
    # calc params
    num_sequences = 0

    # rounds = rounds[:3]

    with open(os.path.join(lstm_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write('{} {} {}\n'.format(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number']))
            for beg, end in r['sequences']:
                expected_frame_count = int((end - beg) / time_step)
                num_sequences += (int(expected_frame_count / frames_per_seq) + 1)

    sequences = []
    for i, r in enumerate(rounds):
        print(i, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
        states = get_player_states(r['id'])
        left_color = r['game']['left_team']['color']
        if left_color == 'W':
            left_color = 'white'
        else:
            left_color = 'nonwhite'

        right_color = r['game']['right_team']['color']
        if right_color == 'W':
            right_color = 'white'
        else:
            right_color = 'nonwhite'
        for beg, end in r['sequences']:
            print(beg, end)
            fvs = FileVideoStream(get_local_path(r), beg + r['begin'], end + r['begin'], time_step).start()
            timepackage.sleep(1.0)
            time = beg
            lookup_time = np.array([time])
            end = end
            data = [[] for _ in range(12)]
            frame_ind = 0
            print('begin main loop')
            begin_time = timepackage.time()
            shape = (frames_per_seq, 67, 67, 3)
            to_predicts = [np.zeros(shape, dtype=np.uint8) for _ in range(12)]
            j = 0
            while True:
                try:
                    frame = fvs.read()
                except Empty:
                    break

                for i in range(12):
                    if i < 6:
                        params = left_params
                        d = look_up_player_state('left', i, time, states)
                        d.update({'color': left_color})
                        x = params['X']
                        x += (params['WIDTH'] + params['MARGIN']) * (i)
                    else:
                        params = right_params
                        d = look_up_player_state('right', i - 6, time, states)
                        d.update({'color': right_color})
                        x = params['X']
                        x += (params['WIDTH'] + params['MARGIN']) * (i - 6)
                    d['rel_time'] = time / (r['end'] - r['begin'])
                    data[i].append(d)
                    box = frame[params['Y']: params['Y'] + params['HEIGHT'],
                          x: x + params['WIDTH']]
                    to_predicts[i][j, ...] = box[None]
                frame_ind += 1
                j += 1
                if j == frames_per_seq:
                    for i in range(12):
                        intermediate_output = embedding_model.predict(to_predicts[i])
                        sequences.append((intermediate_output, data[i]))
                    data = [[] for _ in range(12)]
                    to_predicts = [np.zeros(shape, dtype=np.uint8) for _ in range(12)]
                    j = 0
                time += time_step
                lookup_time = np.array([time])
            if j > 0:
                for i in range(12):
                    intermediate_output = embedding_model.predict(to_predicts[i])
                    sequences.append((intermediate_output, data[i]))
            print('main loop took', timepackage.time() - begin_time)

    random.shuffle(sequences)
    num_sequences = len(sequences)
    num_train = int(num_sequences * 0.8)
    train_sequences = sequences[:num_train]
    num_val = num_sequences - num_train
    val_sequences = sequences[num_train:]
    train_shape = (num_train, frames_per_seq, embedding_size)
    val_shape = (num_val, frames_per_seq, embedding_size)

    print(num_sequences, num_train, num_val)

    hdf5_file = h5py.File(status_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        print(pre, count, shape)
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.float32)
        for k in sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count, frames_per_seq), np.int8)
        hdf5_file.create_dataset("{}_rel_time_label".format(pre), (count, frames_per_seq), np.float32)

    pre = 'train'
    for i, sequence in enumerate(train_sequences):
        if i != 0 and i % 100 == 0:
            print('Train data: {}/{}'.format(i, num_train))
        seq, data = sequence
        for k in range(frames_per_seq):
            if k < len(data):
                d = data[k]
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(d[key])
                hdf5_file['{}_rel_time_label'.format(pre)][i, k] = d['rel_time']
            else:
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(na_lab)
                hdf5_file['{}_rel_time_label'.format(pre)][i, k] = 1.0
        hdf5_file["{}_img".format(pre)][i, ...] = seq[None]

    pre = 'val'
    for i, sequence in enumerate(val_sequences):
        if i != 0 and i % 100 == 0:
            print('Validation data: {}/{}'.format(i, num_val))
        seq, data = sequence
        for k in range(frames_per_seq):
            if k < len(data):
                d = data[k]
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(d[key])
                hdf5_file['{}_rel_time_label'.format(pre)][i, k] = d['rel_time']
            else:
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(na_lab)
                hdf5_file['{}_rel_time_label'.format(pre)][i, k] = 1.0
        hdf5_file["{}_img".format(pre)][i, ...] = seq[None]

    hdf5_file.close()
    new_set_files = {'player': os.path.join(lstm_status_train_dir, 'player_set.txt'),
                     'hero': os.path.join(lstm_status_train_dir, 'hero_set.txt'),
                     'color': os.path.join(lstm_status_train_dir, 'color_set.txt'),
                     'alive': os.path.join(lstm_status_train_dir, 'alive_set.txt'),
                     'ult': os.path.join(lstm_status_train_dir, 'ult_set.txt'),
                     }
    for k, v in new_set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in sets[k]:
                f.write('{}\n'.format(p))


def generate_data_for_mid_lstm(rounds):
    time_step = 0.1
    frames_per_seq = 100
    embedding_size = 50
    import keras
    final_output_weights = os.path.join(cnn_mid_model_dir, 'mid_weights.h5')
    final_output_json = os.path.join(cnn_mid_model_dir, 'mid_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    embedding_model = keras.models.Model(inputs=model.input,
                                         outputs=model.get_layer('representation').output)
    import time as timepackage
    os.makedirs(lstm_mid_train_dir, exist_ok=True)
    status_hd5_path = os.path.join(lstm_mid_train_dir, 'dataset.hdf5')
    if os.path.exists(status_hd5_path):
        print('skipping mid lstm data')
        return
    print('beginning mid lstm data')
    na_lab = 'n/a'
    set_files = {'replay': os.path.join(cnn_mid_train_dir, 'replay_set.txt'),
                 'left_color': os.path.join(cnn_mid_train_dir, 'color_set.txt'),
                 'right_color': os.path.join(cnn_mid_train_dir, 'color_set.txt'),
                 'pause': os.path.join(cnn_mid_train_dir, 'paused_set.txt'),
                 'overtime': os.path.join(cnn_mid_train_dir, 'overtime_set.txt'),
                 'point_status': os.path.join(cnn_mid_train_dir, 'point_set.txt'),
                 }
    sets = {}
    for k, v in set_files.items():
        sets[k] = load_set(v)
        if na_lab not in sets[k]:
            sets[k].append(na_lab)

    params = BOX_PARAMETERS['REGULAR']['MID']
    # calc params
    num_sequences = 0
    for r in rounds:
        expected_frame_count = int((r['end'] - r['begin']) / time_step)
        num_sequences += (int(expected_frame_count / frames_per_seq) + 1)

    sequences = []
    for i, r in enumerate(rounds):
        print(i, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
        states = get_round_states(r['id'])
        left_color = r['game']['left_team']['color']
        if left_color == 'W':
            left_color = 'white'
        else:
            left_color = 'nonwhite'

        right_color = r['game']['right_team']['color']
        if right_color == 'W':
            right_color = 'white'
        else:
            right_color = 'nonwhite'
        fvs = FileVideoStream(get_local_path(r), r['begin'], r['end'], time_step).start()
        timepackage.sleep(1.0)
        time = r['begin']
        data = []
        print('begin main loop')
        begin_time = timepackage.time()
        shape = (frames_per_seq, 140, 300, 3)
        to_predict = np.zeros(shape, dtype=np.uint8)
        j = 0
        while True:
            try:
                frame = fvs.read()
            except Empty:
                break
            d = look_up_round_state(time, states)
            d.update({'left_color': left_color, 'right_color': right_color})
            data.append(d)
            x = params['X']
            box = frame[params['Y']: params['Y'] + params['HEIGHT'],
                  x: x + params['WIDTH']]
            to_predict[j, ...] = box[None]

            j += 1
            time += time_step
            if j == frames_per_seq:
                intermediate_output = embedding_model.predict(to_predict)
                sequences.append((intermediate_output, data))
                data = []
                to_predict = np.zeros(shape, dtype=np.uint8)
                j = 0
        intermediate_output = embedding_model.predict(to_predict)
        sequences.append((intermediate_output, data))
        print('main loop took', timepackage.time() - begin_time)

    num_sequences = len(sequences)
    random.shuffle(sequences)
    num_train = int(num_sequences * 0.8)
    num_val = num_sequences - num_train
    train_sequences = sequences[:num_train]
    val_sequences = sequences[num_train:]
    train_shape = (num_train, frames_per_seq, embedding_size)
    val_shape = (num_val, frames_per_seq, embedding_size)

    hdf5_file = h5py.File(status_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.float32)
        for k in sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count, frames_per_seq), np.int8)

    print(num_sequences, num_train, num_val)

    pre = 'train'
    for i, sequence in enumerate(train_sequences):
        if i != 0 and i % 100 == 0:
            print('Train data: {}/{}'.format(i, num_train))
        seq, data = sequence
        for k in range(frames_per_seq):
            if k < len(data):
                d = data[k]
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(d[key])
            else:
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(na_lab)
        hdf5_file["{}_img".format(pre)][i, ...] = seq[None]

    pre = 'val'
    for i, sequence in enumerate(val_sequences):
        if i != 0 and i % 100 == 0:
            print('Validation data: {}/{}'.format(i, num_val))
        seq, data = sequence
        for k in range(frames_per_seq):
            if k < len(data):
                d = data[k]
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(d[key])
            else:
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(na_lab)
        hdf5_file["{}_img".format(pre)][i, ...] = seq[None]

    hdf5_file.close()
    new_set_files = {'replay': os.path.join(lstm_mid_train_dir, 'replay_set.txt'),
                     'left_color': os.path.join(lstm_mid_train_dir, 'color_set.txt'),
                     'right_color': os.path.join(lstm_mid_train_dir, 'color_set.txt'),
                     'pause': os.path.join(lstm_mid_train_dir, 'paused_set.txt'),
                     'overtime': os.path.join(lstm_mid_train_dir, 'overtime_set.txt'),
                     'point_status': os.path.join(lstm_mid_train_dir, 'point_set.txt'),
                     }
    for k, v in new_set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in sets[k]:
                f.write('{}\n'.format(p))


def generate_data_for_player_cnn(rounds):
    debug = False
    import time as timepackage
    os.makedirs(cnn_status_train_dir, exist_ok=True)
    status_hd5_path = os.path.join(cnn_status_train_dir, 'dataset.hdf5')
    if os.path.exists(status_hd5_path):
        print('skipping player cnn data')
        return
    print('beginning player cnn data')
    time_step = 0.1
    frames_per_seq = 100
    frames = []
    na_lab = 'n/a'
    set_files = {'hero': os.path.join(cnn_status_train_dir, 'hero_set.txt'),
                 'player': os.path.join(cnn_status_train_dir, 'player_set.txt'),
                 'ult': os.path.join(cnn_status_train_dir, 'ult_set.txt'),
                 'alive': os.path.join(cnn_status_train_dir, 'alive_set.txt'),
                 'color': os.path.join(cnn_status_train_dir, 'color_set.txt')}
    sets = {'hero': [na_lab] + HERO_SET,
            'player': [na_lab],
            'ult': [na_lab, 'no_ult', 'has_ult'],
            'alive': [na_lab, 'alive', 'dead'],
            'color': [na_lab]}

    left_params = BOX_PARAMETERS['REGULAR']['LEFT']
    right_params = BOX_PARAMETERS['REGULAR']['RIGHT']
    # rounds = rounds[:3]
    num_frames = 0
    num_sequences = 0
    error_set = {(7432, 59.9),
                 (7492, 59.9),
                 (7088, 239.9),
                 (7456, 169.9),
                 (7088, 219.9),
                 (7024, 109.9),
                 (8007, 109.9),
                 (7015, 39.9),
                 }
    with open(os.path.join(cnn_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            # if r['id'] not in [x[0] for x in error_set]:
            #    continue

            for beg, end in r['sequences']:
                expected_frame_count = int((end - beg) / time_step)
                num_frames += (int(expected_frame_count) + 1)
                num_sequences += (int(expected_frame_count / frames_per_seq) + 1) * 12
    print(num_sequences)

    indexes = random.sample(range(num_sequences), num_sequences)
    num_train = int(num_sequences * 0.8)
    num_val = num_sequences - num_train

    train_shape = (num_train, frames_per_seq, left_params['HEIGHT'], left_params['WIDTH'], 3)
    val_shape = (num_val, frames_per_seq, left_params['HEIGHT'], left_params['WIDTH'], 3)

    hdf5_file = h5py.File(status_hd5_path, mode='w')
    for pre in ['train', 'val']:
        if pre == 'train':
            shape = train_shape
            num = num_train
        else:
            shape = val_shape
            num = num_val
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (num,), np.int32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (num,), np.float32)
        for k in sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (num, frames_per_seq), np.int8)
    # hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

    mean = np.zeros(train_shape[1:], np.float32)
    sequence_ind = 0
    sides = ['left', 'right']
    for round_index, r in enumerate(rounds):
        # if r['id'] not in [x[0] for x in error_set]:
        #    continue
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
        states = get_player_states(r['id'])
        left_color = r['game']['left_team']['color'].lower()
        if left_color not in sets['color']:
            sets['color'].append(left_color)

        right_color = r['game']['right_team']['color'].lower()
        if right_color not in sets['color']:
            sets['color'].append(right_color)

        for beg, end in r['sequences']:
            print(beg, end)
            fvs = FileVideoStream(get_local_path(r), beg + r['begin'], end + r['begin'], time_step,
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
                        data[(side, i)][k] = np.zeros((frames_per_seq,), dtype=np.int8)
            j = 0
            while True:
                try:
                    frame, time = fvs.read()
                except Empty:
                    break
                if sequence_ind >= len(indexes):
                    print('ignoring')
                    sequence_ind += 1
                    continue
                time = round(time, 1)
                for side in ['left', 'right']:
                    if side == 'left':
                        params = left_params
                    else:
                        params = right_params
                    for player_ind in range(6):
                        d = look_up_player_state(side, player_ind, time, states)
                        if side == 'left':
                            d['color'] = left_color
                        else:
                            d['color'] = right_color
                        x = params['X']
                        y = params['Y']
                        x += (params['WIDTH'] + params['MARGIN']) * (player_ind)
                        box = frame[y: y + params['HEIGHT'],
                              x: x + params['WIDTH']]
                        images[(side, player_ind)][j, ...] = box[None]
                        for k, s in sets.items():
                            if d[k] not in s:
                                sets[k].append(d[k])
                            data[(side, player_ind)][k][j] = sets[k].index(d[k])
                        if (r['id'], time) in error_set and debug:  # and sequence_ind % 25 == 0:
                            print(r['id'], time)
                            print(d)
                            cv2.imshow('frame_{}_{}'.format(side, player_ind), box)
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
                            hdf5_file["{}_time_point".format(pre)][index] = time
                            for k, s in sets.items():
                                hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = data[(side, player_ind)][k][None]
                                data[(side, player_ind)][k] = np.zeros((frames_per_seq,), dtype=np.int8)

                            sequence_ind += 1
                    j = 0

                    # if pre == 'train':
                    #    mean += box / num_train
                # if (r['id'], time) in error_set:
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
                        hdf5_file["{}_time_point".format(pre)][index] = time
                        for k, s in sets.items():
                            hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = data[(side, player_ind)][k][None]
                            data[(side, player_ind)][k] = np.zeros((frames_per_seq,), dtype=np.int8)
                        sequence_ind += 1

            print('main loop took', timepackage.time() - begin_time)

    # hdf5_file["train_mean"][...] = mean
    hdf5_file.close()
    for k, v in set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in sets[k]:
                f.write('{}\n'.format(p))


def generate_data_for_mid_cnn(rounds):
    import time as timepackage
    time_step = 0.5
    status_hd5_path = os.path.join(cnn_mid_train_dir, 'dataset.hdf5')
    set_files = {'replay': os.path.join(cnn_mid_train_dir, 'replay_set.txt'),
                 'left_color': os.path.join(cnn_mid_train_dir, 'color_set.txt'),
                 'right_color': os.path.join(cnn_mid_train_dir, 'color_set.txt'),
                 'pause': os.path.join(cnn_mid_train_dir, 'paused_set.txt'),
                 'overtime': os.path.join(cnn_mid_train_dir, 'overtime_set.txt'),
                 'point_status': os.path.join(cnn_mid_train_dir, 'point_set.txt'),
                 }
    na_lab = 'n/a'
    sets = {}
    for k, v in set_files.items():
        sets[k] = []
        if na_lab not in sets[k]:
            sets[k].append(na_lab)
    os.makedirs(cnn_mid_train_dir, exist_ok=True)
    if os.path.exists(status_hd5_path):
        print('skipping mid cnn data')
        return
    print('beginning mid cnn data')

    params = BOX_PARAMETERS['REGULAR']['MID']
    # calc params
    num_frames = 0
    for r in rounds:
        print(r['end'] - r['begin'], time_step)
        expected_frame_count = int((r['end'] - r['begin']) / time_step)
        print(expected_frame_count)
        num_frames += (int(expected_frame_count) + 1)
    print(num_frames)
    indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train

    train_shape = (num_train, params['HEIGHT'], params['WIDTH'], 3)
    val_shape = (num_val, params['HEIGHT'], params['WIDTH'], 3)

    hdf5_file = h5py.File(status_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        for k in sets.keys():
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count,), np.int8)

    hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
    mean = np.zeros(train_shape[1:], np.float32)

    print(num_frames, num_train, num_val)
    frame_ind = 0
    for i, r in enumerate(rounds):
        print(i, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
        states = get_round_states(r['id'])
        left_color = r['game']['left_team']['color']
        if left_color == 'W':
            left_color = 'white'
        else:
            left_color = 'nonwhite'

        right_color = r['game']['right_team']['color']
        if right_color == 'W':
            right_color = 'white'
        else:
            right_color = 'nonwhite'
        fvs = FileVideoStream(get_local_path(r), r['begin'], r['end'], time_step).start()
        timepackage.sleep(1.0)
        time = 0
        print('begin main loop')
        begin_time = timepackage.time()
        while True:
            try:
                frame = fvs.read()
            except Empty:
                break
            index = frame_ind
            adjustment = 0
            print(frame_ind, len(indexes))
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
                print('Train data: {}/{}'.format(index, num_train))
            elif frame_ind != 0 and frame_ind % 100 == 0:
                print('Validation data: {}/{}'.format(index, num_val))
            data = look_up_round_state(time, states)
            data.update({'left_color': left_color, 'right_color': right_color})
            for k, v in sets.items():
                if data[k] not in v:
                    sets[k].append(data[k])
                hdf5_file['{}_{}_label'.format(pre, k)][index] = v.index(data[k])
            x = params['X']
            box = frame[params['Y']: params['Y'] + params['HEIGHT'],
                  x: x + params['WIDTH']]
            hdf5_file["{}_img".format(pre)][index, ...] = box[None]
            if pre == 'train':
                mean += box / num_train
            frame_ind += 1
            time += time_step

        print('main loop took', timepackage.time() - begin_time)

    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()
    for k, v in set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in sets[k]:
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
                {'video_path': get_local_path(r), 'pos': i, 'time_point': t + r['begin'],
                 'first_hero': 'n/a', 'first_color': 'n/a', 'ability': 'n/a', 'headshot': 'n/a', 'second_hero': 'n/a',
                 'second_color': 'n/a', })
    random.shuffle(possible)
    return possible[:number]


def generate_data_for_kf_lstm(rounds):
    slot_based = False
    time_step = 0.1
    frames_per_seq = 100
    embedding_size = 50
    import keras
    final_output_weights = os.path.join(cnn_kf_model_dir, 'kf_weights.h5')
    final_output_json = os.path.join(cnn_kf_model_dir, 'kf_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    embedding_model = keras.models.Model(inputs=model.input,
                                         outputs=model.get_layer('representation').output)
    import time as timepackage
    os.makedirs(lstm_kf_train_dir, exist_ok=True)
    status_hd5_path = os.path.join(lstm_kf_train_dir, 'dataset.hdf5')
    if os.path.exists(status_hd5_path):
        print('skipping kf lstm data')
        return

    print('beginning kf lstm data')

    set_files = {}
    set_files['first_hero'] = os.path.join(cnn_kf_train_dir, 'hero_set.txt')
    set_files['first_color'] = os.path.join(cnn_kf_train_dir, 'color_set.txt')
    set_files['headshot'] = os.path.join(cnn_kf_train_dir, 'headshot_set.txt')
    set_files['ability'] = os.path.join(cnn_kf_train_dir, 'ability_set.txt')
    set_files['second_hero'] = os.path.join(cnn_kf_train_dir, 'hero_set.txt')
    set_files['second_color'] = os.path.join(cnn_kf_train_dir, 'color_set.txt')
    sets = {}
    for k, v in set_files.items():
        if 'hero' in k:
            sets[k] = ['na_lab'] + HERO_SET
        elif k == 'ability':
            sets[k] = ['na_lab'] + ABILITY_SET
        else:
            sets[k] = load_set(v)

    params = BOX_PARAMETERS['REGULAR']['KILL_FEED_SLOT']
    # calc params
    # rounds = rounds[:3]
    num_sequences = 0
    for r in rounds:
        for beg, end in r['sequences']:
            expected_frame_count = int((end - beg) / time_step)
            num_sequences += (int(expected_frame_count / frames_per_seq) + 1)

    sequences = []
    for i, r in enumerate(rounds):
        print(i, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
        events = get_kf_events(r['id'])

        for beg, end in r['sequences']:
            print(beg, end)
            fvs = FileVideoStream(get_local_path(r), beg + r['begin'], end + r['begin'], time_step).start()
            timepackage.sleep(1.0)
            time = beg
            end = end
            frame_ind = 0
            print('begin main loop')
            begin_time = timepackage.time()
            shape = (frames_per_seq, 32, 210, 3)
            to_predicts = [np.zeros(shape, dtype=np.uint8) for _ in range(6)]
            data = [[] for _ in range(6)]
            j = 0
            while True:
                try:
                    frame = fvs.read()
                except Empty:
                    break

                for slot in range(6):
                    x = params['X']
                    y = params['Y']
                    y += (params['HEIGHT'] + params['MARGIN']) * (slot)
                    box = frame[y: y + params['HEIGHT'],
                          x: x + params['WIDTH']]
                    to_predicts[slot][j, ...] = box[None]
                kf, e = construct_kf_at_time(events, time)
                for slot in range(6):
                    if slot > len(kf) - 1:
                        data[slot].append({lab: 'n/a' for lab in
                                           ['first_hero', 'first_color', 'ability', 'headshot', 'second_hero',
                                            'second_color', 'new_event']})
                    else:
                        data[slot].append(kf[slot])

                frame_ind += 1
                j += 1
                time += time_step
                if j == frames_per_seq:
                    if slot_based:
                        for i in range(6):
                            intermediate_output = embedding_model.predict(to_predicts[i])
                            sequences.append((intermediate_output, data[i]))
                    else:
                        img = {}
                        for i in range(6):
                            intermediate_output = embedding_model.predict(to_predicts[i])
                            img[i] = intermediate_output
                        sequences.append((img, data))
                    data = [[] for _ in range(6)]

                    to_predicts = [np.zeros(shape, dtype=np.uint8) for _ in range(6)]
                    j = 0
            if slot_based:
                for i in range(6):
                    intermediate_output = embedding_model.predict(to_predicts[i])
                    sequences.append((intermediate_output, data[i]))
            else:
                img = {}
                for i in range(6):
                    intermediate_output = embedding_model.predict(to_predicts[i])
                    img[i] = intermediate_output
                sequences.append((img, data))
            print('main loop took', timepackage.time() - begin_time)

    num_sequences = len(sequences)
    random.shuffle(sequences)
    num_train = int(num_sequences * 0.8)
    num_val = num_sequences - num_train
    train_sequences = sequences[:num_train]
    val_sequences = sequences[num_train:]
    train_shape = (num_train, frames_per_seq, embedding_size)
    val_shape = (num_val, frames_per_seq, embedding_size)

    hdf5_file = h5py.File(status_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        if slot_based:
            hdf5_file.create_dataset("{}_img".format(pre), shape, np.float32)
            for k in sets.keys():
                hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count, frames_per_seq), np.int8)
        else:
            for slot in range(6):
                hdf5_file.create_dataset("{}_slot_{}_img".format(pre, slot), shape, np.float32)
                for k in sets.keys():
                    hdf5_file.create_dataset("{}_slot_{}_{}_label".format(pre, slot, k), (count, frames_per_seq),
                                             np.int8)

    print(num_sequences, num_train, num_val)

    pre = 'train'
    for i, sequence in enumerate(train_sequences):
        if i != 0 and i % 100 == 0:
            print('Train data: {}/{}'.format(i, num_train))
        seq, data = sequence
        if slot_based:
            for k in range(frames_per_seq):
                if k < len(data):
                    d = data[k]

                    for key, s in sets.items():
                        if d[key] not in s:
                            s.append(d[key])
                        if slot_based:
                            hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(d[key])
                        else:
                            for slot in range(6):
                                hdf5_file['{}_slot_{}_{}_label'.format(pre, slot, key)][i, k] = s.index(d[key])
                else:
                    for key, s in sets.items():
                        if 'n/a' not in s:
                            s.append('n/a')
                        if slot_based:
                            hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index('n/a')
                        else:
                            for slot in range(6):
                                hdf5_file['{}_slot_{}_{}_label'.format(pre, slot, key)][i, k] = s.index('n/a')
        else:
            for slot in range(6):
                slot_data = data[slot]
                for k in range(frames_per_seq):
                    if k < len(slot_data):
                        d = slot_data[k]

                        for key, s in sets.items():
                            if d[key] not in s:
                                s.append(d[key])
                            hdf5_file['{}_slot_{}_{}_label'.format(pre, slot, key)][i, k] = s.index(d[key])
                    else:
                        for key, s in sets.items():
                            if 'n/a' not in s:
                                s.append('n/a')
                            hdf5_file['{}_slot_{}_{}_label'.format(pre, slot, key)][i, k] = s.index('n/a')

        if slot_based:
            hdf5_file["{}_img".format(pre)][i, ...] = seq[None]
        else:
            for slot in range(6):
                hdf5_file["{}_slot_{}_img".format(pre, slot)][i, ...] = seq[slot]

    pre = 'val'
    for i, sequence in enumerate(val_sequences):
        if i != 0 and i % 100 == 0:
            print('Validation data: {}/{}'.format(i, num_val))
        seq, data = sequence

        if slot_based:
            for k in range(frames_per_seq):
                if k < len(data):
                    d = data[k]

                    for key, s in sets.items():
                        if d[key] not in s:
                            s.append(d[key])
                        if slot_based:
                            hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(d[key])
                        else:
                            for slot in range(6):
                                hdf5_file['{}_slot_{}_{}_label'.format(pre, slot, key)][i, k] = s.index(d[key])
                else:
                    for key, s in sets.items():
                        if 'n/a' not in s:
                            s.append('n/a')
                        if slot_based:
                            hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index('n/a')
                        else:
                            for slot in range(6):
                                hdf5_file['{}_slot_{}_{}_label'.format(pre, slot, key)][i, k] = s.index('n/a')
        else:
            for slot in range(6):
                slot_data = data[slot]
                for k in range(frames_per_seq):
                    if k < len(slot_data):
                        d = slot_data[k]

                        for key, s in sets.items():
                            if d[key] not in s:
                                s.append(d[key])
                            hdf5_file['{}_slot_{}_{}_label'.format(pre, slot, key)][i, k] = s.index(d[key])
                    else:
                        for key, s in sets.items():
                            if 'n/a' not in s:
                                s.append('n/a')
                            hdf5_file['{}_slot_{}_{}_label'.format(pre, slot, key)][i, k] = s.index('n/a')
        if slot_based:
            hdf5_file["{}_img".format(pre)][i, ...] = seq[None]
        else:
            for slot in range(6):
                hdf5_file["{}_slot_{}_img".format(pre, slot)][i, ...] = seq[slot]

    hdf5_file.close()
    new_set_files = {'first_hero': os.path.join(lstm_kf_train_dir, 'hero_set.txt'),
                     'first_color': os.path.join(lstm_kf_train_dir, 'color_set.txt'),
                     'ability': os.path.join(lstm_kf_train_dir, 'ability_set.txt'),
                     'headshot': os.path.join(lstm_kf_train_dir, 'headshot_set.txt'),
                     'second_hero': os.path.join(lstm_kf_train_dir, 'hero_set.txt'),
                     'second_color': os.path.join(lstm_kf_train_dir, 'color_set.txt'),
                     }
    for k, v in new_set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in sets[k]:
                f.write('{}\n'.format(p))


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
    #rounds = rounds[:3]
    # rounds = rounds[2:]
    set_files = {'hero': os.path.join(cnn_kf_train_dir, 'hero_set.txt'),
                 'color': os.path.join(cnn_kf_train_dir, 'color_set.txt'),
                 'ability': os.path.join(cnn_kf_train_dir, 'ability_set.txt'),
                 'headshot': os.path.join(cnn_kf_train_dir, 'headshot_set.txt')}
    labels = [na_lab, 'assisting_hero'] + HERO_SET + ABILITY_SET + COLOR_SET
    print(labels)

    labs = ['first_hero', 'first_color', 'ability', 'headshot', 'second_hero', 'second_color']
    time_step = 0.1
    params = BOX_PARAMETERS['REGULAR']['KILL_FEED_SLOT']
    # calc params
    num_frames = 0

    events = {}
    ranges = {}
    with open(os.path.join(cnn_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write(str(r['id']) + '\n')
            events[r['id']] = get_kf_events(r['id'])
            ranges[r['id']] = get_event_ranges(events[r['id']], r['end'] - r['begin'])
            for rd  in ranges[r['id']]:

                expected_duration = rd['end'] - rd['begin']
                expected_frame_count = expected_duration / time_step
                num_frames += (int(expected_frame_count) + 1) * 6

    print(num_frames)
    indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train

    train_shape = (num_train, params['WIDTH'], params['HEIGHT'],  3)
    val_shape = (num_val, params['WIDTH'], params['HEIGHT'], 3)

    hdf5_file = h5py.File(kill_feed_hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.int32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        hdf5_file.create_dataset("{}_label".format(pre), (count, params['WIDTH']), np.int8)

    frame_ind = 0
    for round_index, r in enumerate(rounds):
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
        begin_time = timepackage.time()
        fvs = FileVideoStreamRange(get_local_path(r), r['begin'], ranges[r['id']], time_step).start()
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
                label_data = []
                if slot > len(kf) - 1 or kf[slot]['second_hero'] == na_lab:
                    label_data.append({'begin':0, 'end':params['WIDTH'], 'label': na_lab})
                else:
                    d = kf[slot]
                    second_hero_left, second_hero_right = calculate_hero_boundaries(d['second_player'])

                    ability_left, ability_right = calculate_ability_boundaries(second_hero_left, d['ability'])
                    if d['first_player'] == 'n/a':
                        first_hero_pos = -1
                    else:
                        first_hero_left, first_hero_right = calculate_first_hero_boundaries(ability_left, len(
                            d['assisting_heroes']))
                        first_player_left, first_player_right = calculate_first_player_boundaries(first_hero_left, d['first_player'])
                        first_hero_left = params['WIDTH'] - first_hero_left
                        first_hero_right = params['WIDTH'] - first_hero_right
                        if d['assisting_heroes']:
                            assist_left, assist_right = calculate_assist_boundaries(ability_left, len(
                                d['assisting_heroes']))
                            assist_left = params['WIDTH'] - assist_left
                            assist_right = params['WIDTH'] - assist_right
                            label_data.append({'begin': assist_left, 'end': assist_right, 'label': 'assisting_hero'})

                            #cv2.imshow('frame_assist', box[:,assist_left: assist_right, :])
                    second_hero_left = params['WIDTH'] - second_hero_left
                    second_hero_right = params['WIDTH'] - second_hero_right
                    ability_left = params['WIDTH'] - ability_left
                    ability_right = params['WIDTH'] - ability_right

                    label_data.append({'begin': second_hero_right, 'end': params['WIDTH'], 'label': d['second_color']}) # second name plate
                    label_data.append({'begin': second_hero_left, 'end': second_hero_right, 'label': d['second_hero']})
                    ability = d['ability']
                    if d['headshot']:
                        ability += '_headshot'
                    label_data.append({'begin': ability_left, 'end': ability_right, 'label': ability})
                    label_data.append({'begin': ability_right, 'end': ability_right+3, 'label': d['second_color']})

                    if d['first_player'] != 'n/a':
                        label_data.append({'begin': ability_left-3, 'end': ability_left, 'label': d['first_color']})
                        first_player_left = params['WIDTH'] - first_player_left
                        if first_player_left < 0:
                            first_player_left = 0
                        else:
                            label_data.append({'begin': 0, 'end': first_player_left, 'label': na_lab})
                        first_player_right = params['WIDTH'] - first_player_right
                        if first_player_right > 0:
                            label_data.append({'begin': first_player_left, 'end': first_hero_left, 'label': d['first_color']})
                        if first_hero_left < 0:
                            first_hero_left = 0
                        label_data.append({'begin': first_hero_left, 'end': first_hero_right, 'label': d['first_hero']})


                        #if first_player_left > 0:
                        #    cv2.imshow('frame_na_part', box[:,:first_player_left , :])
                        #cv2.imshow('frame_first_player', box[:,first_player_left: first_player_right, :])
                    label_data = sorted(label_data, key = lambda x: x['begin'])
                    for item in label_data:
                        try:
                            hdf5_file["{}_label".format(pre)][index, item['begin']:item['end']] = labels.index(item['label'])
                        except ValueError:
                            cv2.imshow('frame', box)
                            print(label_data)
                            cv2.waitKey(0)
                            raise

                    #cv2.imshow('frame', box)
                    #cv2.imshow('frame_first_hero', box[:,first_hero_left: first_hero_right, :])
                    #cv2.imshow('frame_second_hero', box[:,second_hero_left: second_hero_right, :])
                    #cv2.imshow('frame_second_player', box[:,second_hero_right: , :])
                    #cv2.imshow('frame_ability'   , box[:,ability_left: ability_right, :])
                    #cv2.waitKey(0)


                #if d['first_hero'] != 'n/a':
                #    cv2.imshow('frame_{}'.format(slot), box)
                #    print(first_hero_left, first_hero_right)
                #    print(second_hero_left, second_hero_right)
                #    print(ability_left, ability_right)
                frame_ind += 1


        print('main loop took', timepackage.time() - begin_time)

    hdf5_file.close()
    with open(os.path.join(cnn_kf_train_dir, 'labels.txt'), 'w', encoding='utf8') as f:
        for p in labels:
            f.write('{}\n'.format(p))


def generate_data_for_cnn(rounds):
    generate_data_for_player_cnn(rounds)
    generate_data_for_kf_cnn(rounds)
    generate_data_for_mid_cnn(rounds)


def generate_data_for_lstm(rounds):
    import time
    begin = time.time()
    generate_data_for_player_lstm(rounds)
    print('generating data for player lstm took ', time.time() - begin)
    begin = time.time()
    generate_data_for_mid_lstm(rounds)
    print('generating data for mid lstm took ', time.time() - begin)
    generate_data_for_kf_lstm(rounds)


def save_round_info(rounds):
    with open(os.path.join(training_data_directory, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write('{} {} {}\n'.format(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number']))


if __name__ == '__main__':
    rounds = get_train_rounds()
    for r in rounds:
        print(r['sequences'])
        local_path = get_local_path(r)
        if local_path is None:
            print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
            get_local_file(r)

    save_round_info(rounds)
    generate_data_for_cnn(rounds)
    # generate_data_for_lstm(rounds)
