import requests
import os
import numpy as np
import h5py
import random
import cv2

local_directory = r'E:\Data\Overwatch\raw_data\annotations\matches'
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

site_url = 'http://localhost:8000/'

api_url = site_url + 'annotator/api/'

BOX_PARAMETERS = {
    'REGULAR': {
        'MID': {
            'HEIGHT': 140,
            'WIDTH': 300,
            'X': 490,
            'Y': 45},

        'KILL_FEED': {
            'Y': 115,
            'X': 1020,
            'WIDTH': 210,
            'HEIGHT': 205
        },
        'KILL_FEED_SLOT': {
            'Y': 115,
            'X': 1020,
            'WIDTH': 210,
            'HEIGHT': 32,
            'MARGIN': 2
        },
        'LEFT': {
            'Y': 40,
            'X': 30,
            'WIDTH': 67,
            'HEIGHT': 67,
            'MARGIN': 4,
        },
        'RIGHT': {
            'Y': 40,
            'X': 830,
            'WIDTH': 67,
            'HEIGHT': 67,
            'MARGIN': 4,
        }
    },
    'APEX': {  # Black borders around video feed
        'MID': {
            'HEIGHT': 140,
            'WIDTH': 300,
            'X': 490,
            'Y': 45},

        'KILL_FEED': {
            'Y': 115,
            'X': 950,
            'WIDTH': 270,
            'HEIGHT': 205
        },
        'LEFT': {
            'Y': 45,
            'X': 51,
            'WIDTH': 67,
            'HEIGHT': 55,
            'MARGIN': 1,
        },
        'RIGHT': {
            'Y': 45,
            'X': 825,
            'WIDTH': 67,
            'HEIGHT': 55,
            'MARGIN': 1,
        }
    }
}


def get_train_rounds():
    url = api_url + 'train_rounds/'
    r = requests.get(url)
    return r.json()


def get_player_states(round_id):
    url = api_url + 'rounds/{}/player_states/'.format(round_id)
    r = requests.get(url)
    data = r.json()
    for side, d in data.items():
        for ind, v in d.items():
            for k in ['ult', 'alive', 'hero']:
                data[side][ind]['{}_array'.format(k)] = np.array([x['end'] for x in v[k]])
    return data


def get_round_states(round_id):
    url = api_url + 'rounds/{}/round_states/'.format(round_id)
    r = requests.get(url)
    return r.json()


def get_kf_events(round_id):
    url = api_url + 'rounds/{}/kill_feed_events/'.format(round_id)
    r = requests.get(url)
    return r.json()


def get_hero_list():
    url = api_url + 'heroes/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_npc_list():
    url = api_url + 'npcs/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_ability_list():
    ability_set = set()
    url = api_url + 'abilities/damaging_abilities/'
    r = requests.get(url)
    resp = r.json()
    for a in resp:
        ability_set.add(a['name'].lower())
    url = api_url + 'abilities/reviving_abilities/'
    r = requests.get(url)
    resp = r.json()
    for a in resp:
        ability_set.add(a['name'].lower())
    return ability_set


HERO_SET = get_hero_list() + get_npc_list()

ABILITY_SET = sorted(get_ability_list())


def get_local_path(r):
    match_directory = os.path.join(local_directory, str(r['game']['match']['wl_id']))
    game_directory = os.path.join(match_directory, str(r['game']['game_number']))
    game_path = os.path.join(game_directory, '{}.mp4'.format(r['game']['game_number']))
    if os.path.exists(game_path):
        return game_path
    match_path = os.path.join(match_directory, '{}.mp4'.format(r['game']['match']['wl_id']))
    if os.path.exists(match_path):
        return match_path


def look_up_player_state(side, index, time, states):
    states = states[side][str(index)]

    data = {}
    ind = np.searchsorted(states['ult_array'], time, side="right")
    if ind == len(states['ult']):
        ind -= 1
    data['ult'] = states['ult'][ind]['status']

    ind = np.searchsorted(states['alive_array'], time, side="right")
    if ind == len(states['alive']):
        ind -= 1
    data['alive'] = states['alive'][ind]['status']

    ind = np.searchsorted(states['hero_array'], time, side="right")
    if ind == len(states['hero']):
        ind -= 1
    data['hero'] = states['hero'][ind]['hero']['name'].lower()

    data['player'] = states['player'].lower()
    return data


def look_up_round_state(time, states):
    data = {}
    for t in states['overtimes']:
        if t['begin'] <= time < t['end']:
            data['overtime'] = t['status']
            break
    else:
        data['overtime'] = states['overtimes'][-1]['status']
    for t in states['pauses']:
        if t['begin'] <= time < t['end']:
            data['pause'] = t['status']
            break
    else:
        data['pause'] = states['pauses'][-1]['status']
    for t in states['replays']:
        if t['begin'] <= time < t['end']:
            data['replay'] = t['status']
            break
    else:
        data['replay'] = states['replays'][-1]['status']
    for t in states['point_status']:
        if t['begin'] <= time < t['end']:
            data['point_status'] = t['status']
            break
    else:
        data['point_status'] = 'n/a'
    return data


def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


from threading import Thread
from queue import Queue, Empty


class FileVideoStream:
    def __init__(self, path, begin, end, time_step, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.stopped = False
        self.begin = begin
        self.end = end
        self.time_step = time_step

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        time_point = self.begin
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                frame_number = int(round(time_point * self.fps)) - 1
                self.stream.set(1, frame_number)
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                # add the frame to the queue
                self.Q.put(frame)
                time_point += self.time_step
                time_point = round(time_point, 1)
                if time_point > self.end:
                    self.stop()
                    return

    def read(self):
        # return next frame in the queue
        if self.stopped and self.Q.qsize() == 0:
            raise Empty
        return self.Q.get()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0


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
    time_step = 0.2
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
            'color': [na_lab, 'white', 'nonwhite']}

    left_params = BOX_PARAMETERS['REGULAR']['LEFT']
    right_params = BOX_PARAMETERS['REGULAR']['RIGHT']
    # rounds = rounds[:3]
    num_frames = 0
    for r in rounds:

        for beg, end in r['sequences']:
            expected_frame_count = int((end - beg) / time_step)
            num_frames += (int(expected_frame_count) + 1) * 12
    print(num_frames)

    indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train

    train_shape = (num_train, left_params['HEIGHT'], left_params['WIDTH'], 3)
    val_shape = (num_val, left_params['HEIGHT'], left_params['WIDTH'], 3)

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
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (num,), np.int8)
    #hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

    mean = np.zeros(train_shape[1:], np.float32)
    #error_set = {(7016, 261.6),
    #             (7319, 442.8),
    #             (7024, 157.8), (7024, 179.2)}
    frame_ind = 0
    for round_index, r in enumerate(rounds):
        #if r['id'] not in [x[0] for x in error_set]:
        #    continue
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
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
            print('begin main loop')
            begin_time = timepackage.time()

            j = 0
            while True:
                try:
                    frame = fvs.read()
                except Empty:
                    break
                if frame_ind >= len(indexes):
                    print('ignoring')
                    frame_ind += 1
                    continue
                time = round(time, 1)
                for side in ['left', 'right']:
                    if side == 'left':
                        params = left_params
                    else:
                        params = right_params
                    for player_ind in range(6):
                        index = indexes[frame_ind]
                        if index < num_train:
                            pre = 'train'
                        else:
                            pre = 'val'
                            index -= num_train
                        if frame_ind != 0 and (frame_ind) % 1000 == 0 and frame_ind < num_train:
                            print('Train data: {}/{}'.format(frame_ind, num_train))
                        elif frame_ind != 0 and frame_ind % 1000 == 0:
                            print('Validation data: {}/{}'.format(frame_ind - num_train, num_val))
                        data = look_up_player_state(side, player_ind, time, states)
                        if side == 'left':
                            data['color'] = left_color
                        else:
                            data['color'] = right_color
                        x = params['X']
                        y = params['Y']
                        x += (params['WIDTH'] + params['MARGIN']) * (player_ind)
                        box = frame[y: y + params['HEIGHT'],
                              x: x + params['WIDTH']]
                        #if (r['id'], time) in error_set:
                        #    print(r['id'], time)
                        #    print(data)
                        #    cv2.imshow('frame', box)
                        #    cv2.waitKey(0)

                        hdf5_file["{}_img".format(pre)][index, ...] = box[None]
                        hdf5_file["{}_round".format(pre)][index] = r['id']
                        hdf5_file["{}_time_point".format(pre)][index] = time
                        for k, s in sets.items():
                            if data[k] not in s:
                                sets[k].append(data[k])
                            hdf5_file['{}_{}_label'.format(pre, k)][index] = sets[k].index(data[k])

                        #if pre == 'train':
                        #    mean += box / num_train
                        frame_ind += 1
                        if debug and j % 25 == 0:
                            print(time, side, player_ind)
                            print(data)
                            cv2.imshow('frame', box)
                            cv2.waitKey(0)
                #if (r['id'], time) in error_set:
                #    error
                time += time_step
                j += 1
            print('main loop took', timepackage.time() - begin_time)

    #hdf5_file["train_mean"][...] = mean
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
        first_color = 'white'
        if e[2].lower() != 'white':
            first_color = 'nonwhite'
        second_color = 'white'
        if e[6].lower() != 'white':
            second_color = 'nonwhite'
        if e[0] > time + 0.25:
            break
        elif e[0] > time:
            possible_kf.insert(0, {'time_point': e[0],
                                   'first_hero': 'n/a', 'first_color': first_color, 'ability': 'n/a', 'headshot': 'n/a',
                                   'second_hero': 'n/a',
                                   'second_color': second_color, })
        if time - window <= e[0] <= time:
            if abs(time - e[0]) < 0.05:
                event_at_time = True
            possible_kf.append({'time_point': e[0],
                                'first_hero': e[1].lower(), 'first_color': first_color, 'ability': e[3].lower(),
                                'headshot': str(e[4]).lower(), 'second_hero': e[5].lower(),
                                'second_color': second_color})
    possible_kf = sorted(possible_kf, key=lambda x: -1 * x['time_point'])
    return possible_kf[:6], event_at_time


def event_at_time(events, time):
    event = {'time_point': time,
             'first_hero': 'n/a', 'first_color': 'n/a', 'ability': 'n/a',
             'headshot': 'n/a', 'second_hero': 'n/a',
             'second_color': 'n/a'}
    for e in events:
        if abs(time - e[0]) < 0.05:
            first_color = 'white'
            if e[2].lower() != 'white':
                first_color = 'nonwhite'
            second_color = 'white'
            if e[6].lower() != 'white':
                second_color = 'nonwhite'
            event = {'time_point': e[0],
                     'first_hero': e[1].lower(), 'first_color': first_color, 'ability': e[3].lower(),
                     'headshot': str(e[4]).lower(), 'second_hero': e[5].lower(),
                     'second_color': second_color}
        if e[0] > time:
            break
    return event


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
    #rounds = rounds[:3]
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
    # rounds = rounds[:3]
    # rounds = rounds[2:]
    set_files = {'hero': os.path.join(cnn_kf_train_dir, 'hero_set.txt'),
                 'color': os.path.join(cnn_kf_train_dir, 'color_set.txt'),
                 'ability': os.path.join(cnn_kf_train_dir, 'ability_set.txt'),
                 'headshot': os.path.join(cnn_kf_train_dir, 'headshot_set.txt')}
    sets = {}
    for k in set_files.keys():
        if k == 'hero':
            sets[k] = [na_lab] + HERO_SET
        elif k == 'ability':
            sets[k] = [na_lab] + ABILITY_SET
        else:
            sets[k] = [na_lab]
    labs = ['first_hero', 'first_color', 'ability', 'headshot', 'second_hero', 'second_color']
    time_step = 1
    params = BOX_PARAMETERS['REGULAR']['KILL_FEED_SLOT']
    # calc params
    num_frames = 0
    for r in rounds:

        for beg, end in r['sequences']:
            expected_frame_count = int((end - beg) / time_step)
            num_frames += (int(expected_frame_count) + 1) * 6
    print(num_frames)
    indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train

    train_shape = (num_train, params['HEIGHT'], params['WIDTH'], 3)
    val_shape = (num_val, params['HEIGHT'], params['WIDTH'], 3)

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
        for k in labs:
            hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count,), np.int8)

    hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
    mean = np.zeros(train_shape[1:], np.float32)
    frame_ind = 0
    for round_index, r in enumerate(rounds):
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
        events = get_kf_events(r['id'])
        # for e in events:
        #    if e[3].lower() == 'coalescence':
        #        print(e)
        #        print(construct_kf_at_time(events, 91))
        #        founderror

        # else:
        #    continue
        # found = False
        for beg, end in r['sequences']:
            print(beg, end)
            fvs = FileVideoStream(get_local_path(r), beg + r['begin'], end + r['begin'], time_step).start()
            timepackage.sleep(1.0)
            time = beg
            print('begin main loop')
            begin_time = timepackage.time()
            j = 0
            while True:
                try:
                    frame = fvs.read()
                except Empty:
                    break
                if frame_ind >= len(indexes):
                    print('ignoring')
                    frame_ind += 1
                    continue
                time = round(time, 1)
                kf, e = construct_kf_at_time(events, time)

                for i in range(6):
                    index = indexes[frame_ind]
                    if index < num_train:
                        pre = 'train'
                    else:
                        pre = 'val'
                        index -= num_train
                    if frame_ind != 0 and (frame_ind) % 1000 == 0 and frame_ind < num_train:
                        print('Train data: {}/{}'.format(frame_ind, num_train))
                    elif frame_ind != 0 and frame_ind % 1000 == 0:
                        print('Validation data: {}/{}'.format(frame_ind - num_train, num_val))

                    x = params['X']
                    y = params['Y']
                    y += (params['HEIGHT'] + params['MARGIN']) * (i)
                    box = frame[y: y + params['HEIGHT'],
                          x: x + params['WIDTH']]
                    if i > len(kf) - 1:
                        d = {lab: 'n/a' for lab in labs}
                    else:
                        d = kf[i]
                    for k in labs:
                        if 'hero' in k:
                            s = sets['hero']
                        elif 'color' in k:
                            s = sets['color']
                        else:
                            s = sets[k]
                        if d[k] not in s:
                            s.append(d[k])
                        hdf5_file['{}_{}_label'.format(pre, k)][index] = s.index(d[k])
                    # if 100 > time > 90:
                    #    print(d)
                    #    print(index)
                    hdf5_file["{}_img".format(pre)][index, ...] = box[None]
                    hdf5_file["{}_round".format(pre)][index] = r['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time
                    if debug and j % 25 == 0:
                        cv2.imshow('frame_{}'.format(i), box)

                    if pre == 'train':
                        mean += box / num_train
                    frame_ind += 1
                if debug and j % 25 == 0:
                    print(time)
                    for k in kf:
                        print(k)
                    cv2.waitKey(0)
                time += time_step
                j += 1
            print('main loop took', timepackage.time() - begin_time)

    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()
    for k, v in set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in sets[k]:
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


def get_local_file(r):
    match_directory = os.path.join(local_directory, str(r['game']['match']['wl_id']))
    game_directory = os.path.join(match_directory, str(r['game']['game_number']))
    import subprocess
    vod_link = r['game']['vod_link']
    match_vod_link = r['game']['match']['vod_link']
    if vod_link == match_vod_link:
        directory = match_directory
        out_template = '{}.%(ext)s'.format(r['game']['match']['wl_id'])
    else:
        directory = game_directory
        out_template = '{}.%(ext)s'.format(r['game']['game_number'])
    print(vod_link)
    if vod_link[0] == 'twitch':
        template = 'https://www.twitch.tv/videos/{}'
    subprocess.call(['youtube-dl', '-F', template.format(vod_link[1]), ], cwd=directory)
    for f in ['720p', '720p30']:
        subprocess.call(['youtube-dl', template.format(vod_link[1]), '-o', out_template, '-f', f], cwd=directory)


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
    generate_data_for_lstm(rounds)
