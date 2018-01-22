import os
import json
import cv2
import numpy as np
import keras

working_dir = r'E:\Data\Overwatch\models'

kf_train_dir = r'E:\Data\Overwatch\training_data\kf_cnn'
player_train_dir = r'E:\Data\Overwatch\training_data\player_status_lstm'
mid_train_dir = r'E:\Data\Overwatch\training_data\mid_cnn'

player_set_files = {
    'hero': os.path.join(player_train_dir, 'hero_set.txt'),
    'color': os.path.join(player_train_dir, 'color_set.txt'),
    'alive': os.path.join(player_train_dir, 'alive_set.txt'),
    'ult': os.path.join(player_train_dir, 'ult_set.txt'),
}
kf_set_files = {
    'first_hero': os.path.join(kf_train_dir, 'hero_set.txt'),
    'first_color': os.path.join(kf_train_dir, 'color_set.txt'),
    'headshot': os.path.join(kf_train_dir, 'headshot_set.txt'),
    'ability': os.path.join(kf_train_dir, 'ability_set.txt'),
    'second_hero': os.path.join(kf_train_dir, 'hero_set.txt'),
    'second_color': os.path.join(kf_train_dir, 'color_set.txt'),

}

mid_set_files = {'replay': os.path.join(mid_train_dir, 'replay_set.txt'),
                 'left_color': os.path.join(mid_train_dir, 'color_set.txt'),
                 'right_color': os.path.join(mid_train_dir, 'color_set.txt'),
                 'pause': os.path.join(mid_train_dir, 'paused_set.txt'),
                 'overtime': os.path.join(mid_train_dir, 'overtime_set.txt'),
                 'point_status': os.path.join(mid_train_dir, 'point_set.txt'),
                 #'map': os.path.join(mid_train_dir, 'map_set.txt'),
                 }


def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


player_sets = {}
for k, v in player_set_files.items():
    player_sets[k] = load_set(v)

kf_sets = {}
for k, v in kf_set_files.items():
    kf_sets[k] = load_set(v)

mid_sets = {}
for k, v in mid_set_files.items():
    mid_sets[k] = load_set(v)

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


def predict_player(frame, player_model):
    left_params = BOX_PARAMETERS['REGULAR']['LEFT']
    right_params = BOX_PARAMETERS['REGULAR']['RIGHT']
    shape = (12, 67, 67, 3)
    to_predict = np.zeros(shape, dtype=np.uint8)
    for i in range(6):
        x = left_params['X']
        x += (left_params['WIDTH'] + left_params['MARGIN']) * i
        box = frame[left_params['Y']: left_params['Y'] + left_params['HEIGHT'],
              x: x + left_params['WIDTH']]
        cv2.imshow('frame_{}_{}'.format('left', i), box)
        to_predict[i, ...] = box[None]
        x = right_params['X']
        x += (right_params['WIDTH'] + right_params['MARGIN']) * i
        box = frame[right_params['Y']: right_params['Y'] + right_params['HEIGHT'],
              x: x + right_params['WIDTH']]
        cv2.imshow('frame_{}_{}'.format('right', i), box)
        to_predict[i + 6, ...] = box[None]
    output = player_model.predict(to_predict)
    output_dict = {}
    for output_ind, lab in enumerate(['hero', 'color', 'alive', 'ult']):
        label_inds = output[output_ind].argmax(axis=1)
        for j in range(label_inds.shape[0]):
            if j < 6:
                side = 'left'
                out_ind = j
            else:
                side = 'right'
                out_ind = j - 6
            if (side, out_ind) not in output_dict:
                output_dict[(side, out_ind)] = {}

            output_dict[(side, out_ind)][lab] = player_sets[lab][label_inds[j]]
    for k, v in output_dict.items():
        print(k)
        print(v)
    cv2.waitKey(0)


def predict_kf(frame, kf_model):
    kf_params = BOX_PARAMETERS['REGULAR']['KILL_FEED_SLOT']
    all_params = BOX_PARAMETERS['REGULAR']['KILL_FEED']
    shape = (6, 32, 210, 3)
    to_predict = np.zeros(shape, dtype=np.uint8)
    kf_box = frame[all_params['Y']: all_params['Y'] + all_params['HEIGHT'],
             all_params['X']: all_params['X'] + all_params['WIDTH']]
    for i in range(6):
        y = kf_params['Y']
        y += (kf_params['HEIGHT'] + kf_params['MARGIN']) * i
        box = frame[y: y + kf_params['HEIGHT'],
              kf_params['X']: kf_params['X'] + kf_params['WIDTH']]
        to_predict[i, ...] = box[None]
    output = kf_model.predict(to_predict)

    output_dict = {}
    for output_ind, lab in enumerate(
            ['first_hero', 'first_color', 'ability', 'headshot', 'second_hero', 'second_color']):
        label_inds = output[output_ind].argmax(axis=1)
        output_dict['{}'.format(lab)] = []
        for j in range(label_inds.shape[0]):
            output_dict['{}'.format(lab)].append(kf_sets[lab][label_inds[j]])
    kf = []
    for f in range(6):
        slot = {}
        for k, v in output_dict.items():
            slot[k] = v[f]
        if all(x == 'n/a' for x in slot.values()):
            continue
        kf.append(list(slot.values()))

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

def predict_on_video(video_path, begin, end):
    print('beginning prediction')
    time_step = 0.1
    frames = (end - begin) / time_step
    frames_per_seq = 200
    shape = (frames_per_seq, 67, 67, 3)
    num_players = 1
    fvs = FileVideoStream(video_path, begin, end, time_step).start()
    left_params = BOX_PARAMETERS['REGULAR']['LEFT']
    right_params = BOX_PARAMETERS['REGULAR']['RIGHT']
    j = 0
    statuses = {}
    for side in ['left', 'right']:
        for i in range(6):
            statuses[(side, i)] = {k: [] for k in player_sets.keys()}
    to_predicts = {k: np.zeros(shape, dtype=np.uint8) for k in statuses.keys()}
    expected_sequence_count = int(frames/ frames_per_seq) + 1
    time = 0
    seq_ind = 0
    print(begin, end, end - begin)
    frame_ind = 0
    while True:
        try:
            frame = fvs.read()
        except Empty:
            break
        print('frames', frame_ind, frames)
        frame_ind += 1
        for player in statuses.keys():
            side, pos = player
            if side == 'left':
                params = left_params
            else:
                params = right_params
            x = params['X']
            x += (params['WIDTH'] + params['MARGIN']) * (pos)

            box = frame[params['Y']: params['Y'] + params['HEIGHT'],
                  x: x + params['WIDTH']]
            to_predicts[player][j, ...] = box[None]
        j += 1
        if j == frames_per_seq:
            print('sequences', seq_ind, expected_sequence_count)
            seq_ind += 1
            for player in statuses.keys():
                intermediate_output = player_cnn_model.predict(to_predicts[player])
                output = player_lstm_model.predict(intermediate_output[None])
                for output_ind, (output_key, s) in enumerate(player_sets.items()):
                    label_inds = output[output_ind].argmax(axis=2)
                    for t_ind in range(label_inds.shape[1]):
                        current_time = time + (t_ind * time_step)
                        label = s[label_inds[0,t_ind]]
                        if len(statuses[player][output_key]) == 0:
                            statuses[player][output_key].append({'begin': 0, 'end':0, 'status':label})
                        else:
                            if label == statuses[player][output_key][-1]['status']:
                                statuses[player][output_key][-1]['end'] = current_time
                            else:
                                statuses[player][output_key].append({'begin': current_time, 'end': current_time, 'status':label})

            time += (j * time_step)
            print(time, end - begin)
            to_predicts = {k: np.zeros(shape, dtype=np.uint8) for k in statuses.keys()}
            j = 0
    print(statuses)

def load_player_cnn_model():
    final_output_weights = os.path.join(working_dir, 'player_status_cnn', 'player_weights.h5')
    final_output_json = os.path.join(working_dir, 'player_status_cnn', 'player_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    embedding_model = keras.models.Model(inputs=model.input,
                                         outputs=model.get_layer('representation').output)
    return embedding_model


def load_player_lstm_model():
    final_output_weights = os.path.join(working_dir, 'player_status_lstm', 'player_weights.h5')
    final_output_json = os.path.join(working_dir, 'player_status_lstm', 'player_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    return model


def load_kf_model():
    final_output_weights = os.path.join(working_dir, 'kf_weights.h5')
    final_output_json = os.path.join(working_dir, 'kf_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    return model


if __name__ == '__main__':
    # kf_model = load_kf_model()
    player_cnn_model = load_player_cnn_model()
    player_lstm_model = load_player_lstm_model()

    print('loaded model')
    to_annotate = r"E:\Data\Overwatch\raw_data\annotations\matches\2372\1\1.mp4"
    begin, end = 550, 973
    #to_annotate = r"E:\Data\Overwatch\raw_data\annotations\matches\2360\2360.mp4"
    #begin, end = 2000, 2100
    predict_on_video(to_annotate, begin, end)
