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
    return r.json()


def get_round_states(round_id):
    url = api_url + 'rounds/{}/round_states/'.format(round_id)
    r = requests.get(url)
    return r.json()


def get_kf_events(round_id):
    url = api_url + 'rounds/{}/kill_feed_events/'.format(round_id)
    r = requests.get(url)
    return r.json()


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
    for t in states['ult']:
        if t['begin'] <= time < t['end']:
            data['ult'] = t['status']
            break
    for t in states['alive']:
        if t['begin'] <= time < t['end']:
            data['alive'] = t['status']
            break
    for t in states['hero']:
        if t['begin'] <= time < t['end']:
            data['hero'] = t['hero']['name'].lower()
            break
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
                self.stream.set(1, int(time_point * self.fps))
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                # add the frame to the queue
                self.Q.put(frame)
                time_point += self.time_step
                if time_point > self.end:
                    self.stop()
                    return

    def read(self):
        # return next frame in the queue
        if self.stopped:
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
    frames_per_seq = 200
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
        sets[k] = load_set(v)
        if na_lab not in sets[k]:
            sets[k].append(na_lab)

    left_params = BOX_PARAMETERS['REGULAR']['LEFT']
    right_params = BOX_PARAMETERS['REGULAR']['RIGHT']
    # calc params
    num_sequences = 0

    #rounds = rounds[:3]

    with open(os.path.join(lstm_status_train_dir, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write('{} {} {}\n'.format(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number']))
            for beg, end in r['sequences']:
                expected_frame_count = int((end - beg) /time_step)
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
            end = end
            data = [[] for _ in range(12)]
            frame_ind = 0
            print('begin main loop')
            begin_time = timepackage.time()
            shape = (200, 67, 67, 3)
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
                        d = look_up_player_state('right', i-6, time, states)
                        d.update({'color': right_color})
                        x = params['X']
                        x += (params['WIDTH'] + params['MARGIN']) * (i-6)
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

    pre = 'train'
    for i, sequence in enumerate(train_sequences):
        if i != 0 and i % 100 == 0:
            print('Train data: {}/{}'.format(i, num_train))
        seq, data = sequence
        for k in range(frames_per_seq):
            if k < len(data) - 1:
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
            if k < len(data) - 1:
                d = data[k]
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(d[key])
            else:
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(na_lab)
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
    frames_per_seq = 200
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
        expected_frame_count = int((r['end'] - r['begin']) /time_step)
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
        shape = (200, 140, 300, 3)
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
            if k < len(data) - 1:
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
            if k < len(data) - 1:
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
    import time as timepackage
    os.makedirs(cnn_status_train_dir, exist_ok=True)
    status_hd5_path = os.path.join(cnn_status_train_dir, 'dataset.hdf5')
    if os.path.exists(status_hd5_path):
        print('skipping player cnn data')
        return
    print('beginning player cnn data')
    time_step = 0.5
    frames = []
    na_lab = 'n/a'
    hero_set = [na_lab]
    player_set = []
    ult_set = ['no_ult', 'has_ult']
    alive_set = ['alive', 'dead']
    color_set = []
    for r in rounds:
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
        states = get_player_states(r['id'])
        left_color = r['game']['left_team']['color']
        if left_color == 'W':
            left_color = 'white'
        else:
            left_color = 'nonwhite'
        if left_color not in color_set:
            color_set.append(left_color)

        right_color = r['game']['right_team']['color']
        if right_color == 'W':
            right_color = 'white'
        else:
            right_color = 'nonwhite'
        if right_color not in color_set:
            color_set.append(right_color)
        for beg, end in r['sequences']:
            time = beg
            end = end
            while time < end:
                actual_time = time + r['begin']
                for i in range(6):
                    data = look_up_player_state('left', i, time, states)
                    data.update({'time_point': actual_time, 'video_path': get_local_path(r), 'pos': i, 'side': 'left',
                                 'round_time': timepackage.strftime('%M:%S', timepackage.gmtime(time)),
                                 'color': left_color})
                    if data['hero'] not in hero_set:
                        hero_set.append(data['hero'])
                    if data['player'] not in player_set:
                        player_set.append(data['player'])
                    frames.append(data)

                    data = look_up_player_state('right', i, time, states)
                    data.update({'time_point': actual_time, 'video_path': get_local_path(r), 'pos': i, 'side': 'right',
                                 'round_time': timepackage.strftime('%M:%S', timepackage.gmtime(time)),
                                 'color': right_color})
                    if data['hero'] not in hero_set:
                        hero_set.append(data['hero'])
                    if data['player'] not in player_set:
                        player_set.append(data['player'])
                    frames.append(data)

                time += time_step

    hero_set = sorted(hero_set)
    player_set = sorted(player_set)
    ult_set = sorted(ult_set)
    alive_set = sorted(alive_set)
    color_set = sorted(color_set)

    num_frames = len(frames)
    random.shuffle(frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train
    train_frames = frames[:num_train]
    val_frames = frames[num_train:]
    print(len(train_frames), len(val_frames))
    left_params = BOX_PARAMETERS['REGULAR']['LEFT']
    right_params = BOX_PARAMETERS['REGULAR']['RIGHT']
    train_shape = (num_train, left_params['HEIGHT'], left_params['WIDTH'], 3)
    val_shape = (num_val, left_params['HEIGHT'], left_params['WIDTH'], 3)

    hdf5_file = h5py.File(status_hd5_path, mode='w')
    hdf5_file.create_dataset("train_img", train_shape, np.uint8)
    hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
    hdf5_file.create_dataset("train_player_label", (num_train,), np.int8)
    hdf5_file.create_dataset("train_hero_label", (num_train,), np.int8)
    hdf5_file.create_dataset("train_ult_label", (num_train,), np.int8)
    hdf5_file.create_dataset("train_alive_label", (num_train,), np.int8)
    hdf5_file.create_dataset("train_color_label", (num_train,), np.int8)

    hdf5_file.create_dataset("val_img", val_shape, np.uint8)
    hdf5_file.create_dataset("val_player_label", (num_val,), np.int8)
    hdf5_file.create_dataset("val_hero_label", (num_val,), np.int8)
    hdf5_file.create_dataset("val_ult_label", (num_val,), np.int8)
    hdf5_file.create_dataset("val_alive_label", (num_val,), np.int8)
    hdf5_file.create_dataset("val_color_label", (num_val,), np.int8)

    mean = np.zeros(train_shape[1:], np.float32)
    caps = {}
    hero_set_file = os.path.join(cnn_status_train_dir, 'hero_set.txt')
    color_set_file = os.path.join(cnn_status_train_dir, 'color_set.txt')
    player_set_file = os.path.join(cnn_status_train_dir, 'player_set.txt')
    ult_set_file = os.path.join(cnn_status_train_dir, 'ult_set.txt')
    alive_set_file = os.path.join(cnn_status_train_dir, 'alive_set.txt')

    for i, f in enumerate(train_frames):
        if i % 100 == 0 and i > 1:
            print('Train data: {}/{}'.format(i, num_train))
        if f['video_path'] not in caps:
            caps[f['video_path']] = cv2.VideoCapture(f['video_path'])
        fps = caps[f['video_path']].get(cv2.CAP_PROP_FPS)
        caps[f['video_path']].set(1, int(f['time_point'] * fps))
        ret, frame = caps[f['video_path']].read()
        if f['side'] == 'left':
            params = left_params
        else:
            params = right_params
        x = params['X']
        x += (params['WIDTH'] + params['MARGIN']) * (f['pos'])
        box = frame[params['Y']: params['Y'] + params['HEIGHT'],
              x: x + params['WIDTH']]
        # print(f)
        # cv2.imshow('frame', box)
        # cv2.waitKey(0)
        hdf5_file["train_img"][i, ...] = box[None]
        mean += box / len(train_frames)

        hdf5_file['train_player_label'][i] = player_set.index(f['player'])
        hdf5_file['train_hero_label'][i] = hero_set.index(f['hero'])
        hdf5_file['train_ult_label'][i] = ult_set.index(f['ult'])
        hdf5_file['train_alive_label'][i] = alive_set.index(f['alive'])
        hdf5_file['train_color_label'][i] = color_set.index(f['color'])

    for i, f in enumerate(val_frames):
        if i % 100 == 0 and i > 1:
            print('Validation data: {}/{}'.format(i, num_val))
        if f['video_path'] not in caps:
            caps[f['video_path']] = cv2.VideoCapture(f['video_path'])
        fps = caps[f['video_path']].get(cv2.CAP_PROP_FPS)
        caps[f['video_path']].set(1, int(f['time_point'] * fps))
        ret, frame = caps[f['video_path']].read()
        if f['side'] == 'left':
            params = left_params
        else:
            params = right_params
        x = params['X']
        x += (params['WIDTH'] + params['MARGIN']) * (f['pos'])
        box = frame[params['Y']: params['Y'] + params['HEIGHT'],
              x: x + params['WIDTH']]
        hdf5_file["val_img"][i, ...] = box[None]

        hdf5_file['val_player_label'][i] = player_set.index(f['player'])
        hdf5_file['val_hero_label'][i] = hero_set.index(f['hero'])
        hdf5_file['val_ult_label'][i] = ult_set.index(f['ult'])
        hdf5_file['val_alive_label'][i] = alive_set.index(f['alive'])
        hdf5_file['val_color_label'][i] = color_set.index(f['color'])

    for v in caps.values():
        v.release()

    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()
    with open(hero_set_file, 'w', encoding='utf8') as f:
        for p in hero_set:
            f.write('{}\n'.format(p))
    with open(player_set_file, 'w', encoding='utf8') as f:
        for p in player_set:
            f.write('{}\n'.format(p))
    with open(ult_set_file, 'w', encoding='utf8') as f:
        for p in ult_set:
            f.write('{}\n'.format(p))
    with open(alive_set_file, 'w', encoding='utf8') as f:
        for p in alive_set:
            f.write('{}\n'.format(p))
    with open(color_set_file, 'w', encoding='utf8') as f:
        for p in color_set:
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
        print(r['end']- r['begin'], time_step)
        expected_frame_count = int((r['end'] - r['begin']) /time_step)
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
            for k,v in sets.items():
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
    window = 7
    possible_kf = []
    event_at_time = False
    for e in events:
        if e[0] > time+0.2:
            break
        elif e[0] > time:
            possible_kf.insert(0,{'time_point': e[0],
                 'first_hero': 'n/a', 'first_color': e[2], 'ability': 'n/a', 'headshot': 'n/a', 'second_hero': 'n/a',
                 'second_color': e[-1], })
        if time - window <= e[0] <= time:
            if abs(time - e[0]) < 0.05:
                event_at_time = True
            possible_kf.append({'time_point': e[0],
                 'first_hero': e[1], 'first_color': e[2], 'ability': e[3], 'headshot': e[4], 'second_hero': e[5],
                 'second_color': e[6]})
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
    time_step = 0.1
    frames_per_seq = 200
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

    set_files = {'first_hero': os.path.join(cnn_kf_train_dir, 'hero_set.txt'),
             'first_color': os.path.join(cnn_kf_train_dir, 'color_set.txt'),
             'ability': os.path.join(cnn_kf_train_dir, 'ability_set.txt'),
             'headshgot': os.path.join(cnn_kf_train_dir, 'headshot_set.txt'),
             'second_hero': os.path.join(cnn_kf_train_dir, 'hero_set.txt'),
             'second_color': os.path.join(cnn_kf_train_dir, 'color_set.txt'),
                 }
    sets = {}
    for k, v in set_files.items():
        sets[k] = load_set(v)
    sets['new_event'] = ['no', 'yes']
    params = BOX_PARAMETERS['REGULAR']['KILL_FEED_SLOT']
    # calc params
    num_sequences = 0
    for r in rounds:
        for beg, end in r['sequences']:
            expected_frame_count = int((end - beg) /time_step)
            num_sequences += (int(expected_frame_count / frames_per_seq) + 1)

    sequences =[]
    for i, r in enumerate(rounds):
        print(i, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
        events = get_kf_events(r['id'])

        for beg, end in r['sequences']:
            fvs = FileVideoStream(get_local_path(r), beg + r['begin'], end + r['begin'], time_step).start()
            timepackage.sleep(1.0)
            time = beg
            end = end
            data = []
            frame_ind = 0
            print('begin main loop')
            begin_time = timepackage.time()
            shape = (200, 67, 67, 3)
            to_predicts = [np.zeros(shape, dtype=np.uint8) for _ in range(6)]
            j = 0
            while True:
                try:
                    frame = fvs.read()
                except Empty:
                    break

                kf, e = construct_kf_at_time(events, time)
                d = {'new_event': 'no'}
                if e:
                    d['new_event'] = 'yes'
                for i in range(6):
                    x = params['X']
                    y = params['Y']
                    y += (params['HEIGHT'] + params['MARGIN']) * (i)
                    box = frame[y: y + params['HEIGHT'],
                          x: x + params['WIDTH']]
                    to_predicts[i][j, ...] = box[None]
                    for lab in ['first_hero', 'first_color', 'ability', 'headshot', 'second_hero', 'second_color']:
                        if i > len(kf) - 1:
                            d['slot_{}_{}'.format(i, lab)] = 'n/a'
                        else:
                            d['slot_{}_{}'.format(i, lab)] = kf[i][lab]
                data.append(d)
                frame_ind += 1
                j += 1
                if j == frames_per_seq:

                    for i in range(6):
                        intermediate_output = embedding_model.predict(to_predicts[i])
                        sequences.append((intermediate_output, data))
                    data = []
                    to_predicts = [np.zeros(shape, dtype=np.uint8) for _ in range(6)]
                    j = 0

            for i in range(6):
                intermediate_output = embedding_model.predict(to_predicts[i])
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
            if k < len(data) - 1:
                d = data[k]
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(d[key])
            else:
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index('n/a')
        hdf5_file["{}_img".format(pre)][i, ...] = seq[None]

    pre = 'val'
    for i, sequence in enumerate(val_sequences):
        if i != 0 and i % 100 == 0:
            print('Validation data: {}/{}'.format(i, num_val))
        seq, data = sequence
        for k in range(frames_per_seq):
            if k < len(data) - 1:
                d = data[k]
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index(d[key])
            else:
                for key, s in sets.items():
                    hdf5_file['{}_{}_label'.format(pre, key)][i, k] = s.index('n/a')
        hdf5_file["{}_img".format(pre)][i, ...] = seq[None]

    hdf5_file.close()
    new_set_files = {'first_hero': os.path.join(cnn_kf_train_dir, 'hero_set.txt'),
             'first_color': os.path.join(cnn_kf_train_dir, 'color_set.txt'),
             'ability': os.path.join(cnn_kf_train_dir, 'ability_set.txt'),
             'headshgot': os.path.join(cnn_kf_train_dir, 'headshot_set.txt'),
             'second_hero': os.path.join(cnn_kf_train_dir, 'hero_set.txt'),
             'second_color': os.path.join(cnn_kf_train_dir, 'color_set.txt'),
             'new_event': os.path.join(lstm_kf_train_dir, 'new_event_set.txt'),
                 }
    for k, v in new_set_files.items():
        with open(v, 'w', encoding='utf8') as f:
            for p in sets[k]:
                f.write('{}\n'.format(p))


def generate_data_for_kf_cnn(rounds):
    kill_feed_hd5_path = os.path.join(cnn_kf_train_dir, 'dataset.hdf5')
    os.makedirs(cnn_kf_train_dir, exist_ok=True)
    if os.path.exists(kill_feed_hd5_path):
        print('skipping kf cnn data')
        return
    print('beginning kf cnn data')
    na_lab = 'n/a'
    hero_set = set([na_lab])
    color_set = set([na_lab])
    ability_set = set([na_lab])
    headshot_set = set([na_lab])
    frames = []
    hero_set = set([na_lab])
    color_set = set([na_lab])
    ability_set = set([na_lab])
    headshot_set = set([na_lab])
    frames = []
    hero_set_file = os.path.join(cnn_kf_train_dir, 'hero_set.txt')
    color_set_file = os.path.join(cnn_kf_train_dir, 'color_set.txt')
    ability_set_file = os.path.join(cnn_kf_train_dir, 'ability_set.txt')
    headshot_set_file = os.path.join(cnn_kf_train_dir, 'headshot_set.txt')
    usable_event_count = 0
    for i, r in enumerate(rounds):
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
        events = get_kf_events(r['id'])
        time = 0
        for i, e in enumerate(events):
            actual_time = e[0] + r['begin']
            if i != 0 and e[0] - events[i - 1][0] < 0.1:
                continue
            if i < len(events) - 1 and events[i + 1][0] - e[0] < 0.1:
                continue
            # figure out position of event
            pos = 0
            if i < len(events) - 1:
                j = i + 1

                while True:
                    next_time_slot = events[j][0]
                    if next_time_slot - e[0] > 0.25:
                        break
                    j += 1
                    if j > len(events) - 1:
                        break
                    pos += 1
            first_hero = e[1].lower()
            first_color = e[2].lower()
            if first_color != 'white':
                first_color = 'nonwhite'
            ability = e[3].lower()
            headshot = str(e[4])
            second_hero = e[5].lower()
            second_color = e[6].lower()
            if second_color != 'white':
                second_color = 'nonwhite'
            hero_set.add(first_hero)
            color_set.add(first_color)
            ability_set.add(ability)
            headshot_set.add(headshot)
            hero_set.add(second_hero)
            color_set.add(second_color)
            time_point = actual_time
            frames.append({'time_point': time_point, 'pos': pos, 'first_hero': first_hero, 'first_color': first_color,
                           'ability': ability,
                           'headshot': headshot, 'second_hero': second_hero, 'second_color': second_color,
                           'video_path': get_local_path(r)})
            time_point += 1 / 30
            frames.append({'time_point': time_point, 'pos': pos, 'first_hero': first_hero, 'first_color': first_color,
                           'ability': ability,
                           'headshot': headshot, 'second_hero': second_hero, 'second_color': second_color,
                           'video_path': get_local_path(r)})
            time_point += 1 / 30
            frames.append({'time_point': time_point, 'pos': pos, 'first_hero': first_hero, 'first_color': first_color,
                           'ability': ability,
                           'headshot': headshot, 'second_hero': second_hero, 'second_color': second_color,
                           'video_path': get_local_path(r)})
            usable_event_count += 3
        frames.extend(generate_negative_kf_examples(r, events, int(usable_event_count / 3)))
    hero_set = sorted(hero_set)
    color_set = sorted(color_set)
    ability_set = sorted(ability_set)
    headshot_set = sorted(headshot_set)

    num_train_events = int(len(frames) * 0.8)
    num_val_events = len(frames) - num_train_events
    random.shuffle(frames)
    train_events = frames[:num_train_events]
    val_events = frames[num_train_events:]
    params = BOX_PARAMETERS['REGULAR']['KILL_FEED_SLOT']
    train_shape = (num_train_events, 32, 210, 3)
    val_shape = (num_val_events, 32, 210, 3)
    hdf5_file = h5py.File(kill_feed_hd5_path, mode='w')
    hdf5_file.create_dataset("train_img", train_shape, np.uint8)
    hdf5_file.create_dataset("val_img", val_shape, np.uint8)
    hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
    labs = ['first_hero', 'first_color', 'ability', 'headshot', 'second_hero', 'second_color']
    for lab in labs:
        hdf5_file.create_dataset("train_{}_label".format(lab), (num_train_events,), np.int8)
        hdf5_file.create_dataset("val_{}_label".format(lab), (num_val_events,), np.int8)
    mean = np.zeros(train_shape[1:], np.float32)
    print(num_train_events, num_val_events)
    caps = {}
    for i, d in enumerate(train_events):
        if i % 100 == 0 and i > 1:
            print('Train data: {}/{}'.format(i, num_train_events))
        if d['video_path'] not in caps:
            caps[d['video_path']] = cv2.VideoCapture(d['video_path'])
        cap = caps[d['video_path']]
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(d['time_point'] * fps) + 1
        cap.set(1, frame_number)
        ret, frame = cap.read()
        y = params['Y']
        y += (params['HEIGHT'] + params['MARGIN']) * d['pos']
        box = frame[y: y + params['HEIGHT'],
              params['X']: params['X'] + params['WIDTH']]
        # print(d)
        # cv2.imshow('frame', box)
        # cv2.waitKey(0)
        hdf5_file["train_img"][i, ...] = box[None]
        mean += box / len(train_events)
        hdf5_file['train_first_hero_label'][i] = hero_set.index(d['first_hero'])
        hdf5_file['train_first_color_label'][i] = color_set.index(d['first_color'])
        hdf5_file['train_ability_label'][i] = ability_set.index(d['ability'])
        hdf5_file['train_headshot_label'][i] = headshot_set.index(d['headshot'])
        hdf5_file['train_second_hero_label'][i] = hero_set.index(d['second_hero'])
        hdf5_file['train_second_color_label'][i] = color_set.index(d['second_color'])

    for i, d in enumerate(val_events):
        if i % 100 == 0 and i > 1:
            print('Validation data: {}/{}'.format(i, num_val_events))
        if d['video_path'] not in caps:
            caps[d['video_path']] = cv2.VideoCapture(d['video_path'])
        cap = caps[d['video_path']]
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(d['time_point'] * fps) + 1
        cap.set(1, frame_number)
        ret, frame = cap.read()
        y = params['Y']
        y += (params['HEIGHT'] + params['MARGIN']) * d['pos']
        box = frame[y: y + params['HEIGHT'],
              params['X']: params['X'] + params['WIDTH']]
        # box = cv2.resize(box, (128, 32))
        # box = cv2.copyMakeBorder(box, 48, 48, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        hdf5_file["val_img"][i, ...] = box[None]
        hdf5_file['val_first_hero_label'][i] = hero_set.index(d['first_hero'])
        hdf5_file['val_first_color_label'][i] = color_set.index(d['first_color'])
        hdf5_file['val_ability_label'][i] = ability_set.index(d['ability'])
        hdf5_file['val_headshot_label'][i] = headshot_set.index(d['headshot'])
        hdf5_file['val_second_hero_label'][i] = hero_set.index(d['second_hero'])
        hdf5_file['val_second_color_label'][i] = color_set.index(d['second_color'])

    hdf5_file["train_mean"][...] = mean
    hdf5_file.close()
    with open(hero_set_file, 'w', encoding='utf8') as f:
        for p in hero_set:
            f.write('{}\n'.format(p))
    with open(color_set_file, 'w', encoding='utf8') as f:
        for p in color_set:
            f.write('{}\n'.format(p))
    with open(ability_set_file, 'w', encoding='utf8') as f:
        for p in ability_set:
            f.write('{}\n'.format(p))
    with open(headshot_set_file, 'w', encoding='utf8') as f:
        for p in headshot_set:
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
    subprocess.call(['youtube-dl', template.format(vod_link[1]), '-o', out_template, '-f', '720p'], cwd=directory)

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
