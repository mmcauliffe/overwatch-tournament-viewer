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
                 # 'map': os.path.join(mid_train_dir, 'map_set.txt'),
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
#for k, v in kf_set_files.items():
#    kf_sets[k] = load_set(v)

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
        if self.stopped and not self.more():
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
    frames_per_seq = 100
    shape = (frames_per_seq, 67, 67, 3)
    kf_shape = (frames_per_seq, 32, 210, 3)
    num_players = 1
    fvs = FileVideoStream(video_path, begin, end, time_step).start()
    left_params = BOX_PARAMETERS['REGULAR']['LEFT']
    right_params = BOX_PARAMETERS['REGULAR']['RIGHT']
    kf_params = BOX_PARAMETERS['REGULAR']['KILL_FEED_SLOT']
    j = 0
    lstm_kfs = {}
    # cnn_kfs = {}
    for i in range(6):
        lstm_kfs[i] = {k: [] for k in kf_sets.keys()}
        # cnn_kfs[i] = {k: [] for k in kf_sets.keys()}
    lstm_statuses = {}
    cnn_statuses = {}
    for side in ['left', 'right']:
        for i in range(6):
            lstm_statuses[(side, i)] = {k: [] for k in player_sets.keys()}
            cnn_statuses[(side, i)] = {k: [] for k in player_sets.keys()}
    to_predicts = {k: np.zeros(shape, dtype=np.uint8) for k in lstm_statuses.keys()}
    to_predicts_kf = {k: np.zeros(kf_shape, dtype=np.uint8) for k in lstm_kfs.keys()}
    expected_sequence_count = int(frames / frames_per_seq) + 1
    time = 0
    seq_ind = 0
    print(begin, end, end - begin)
    frame_ind = 0
    kf = []
    while True:
        try:
            frame = fvs.read()
        except Empty:
            break
        print('frames', frame_ind, frames)
        frame_ind += 1
        for player in lstm_statuses.keys():
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
        for slot in lstm_kfs.keys():
            x = kf_params['X']
            y = kf_params['Y']
            y += (kf_params['HEIGHT'] + kf_params['MARGIN']) * (slot)
            box = frame[y: y + kf_params['HEIGHT'],
                  x: x + kf_params['WIDTH']]
            to_predicts_kf[slot][j, ...] = box[None]
        j += 1
        if j == frames_per_seq:
            print('sequences', seq_ind, expected_sequence_count)
            seq_ind += 1
            for player in lstm_statuses.keys():

                intermediate_output = player_cnn_intermediate_model.predict(to_predicts[player])
                lstm_output = player_lstm_model.predict(intermediate_output[None])
                cnn_output = player_cnn_model.predict(to_predicts[player])
                for output_ind, (output_key, s) in enumerate(player_sets.items()):
                    cnn_inds = cnn_output[output_ind].argmax(axis=1)
                    label_inds = lstm_output[output_ind].argmax(axis=2)
                    for t_ind in range(frames_per_seq):
                        current_time = time + (t_ind * time_step)
                        lstm_label = s[label_inds[0, t_ind]]
                        if lstm_label != 'n/a':
                            if len(lstm_statuses[player][output_key]) == 0:
                                lstm_statuses[player][output_key].append({'begin': 0, 'end': 0, 'status': lstm_label})
                            else:
                                if lstm_label == lstm_statuses[player][output_key][-1]['status']:
                                    lstm_statuses[player][output_key][-1]['end'] = current_time
                                else:
                                    lstm_statuses[player][output_key].append(
                                        {'begin': current_time, 'end': current_time, 'status': lstm_label})
                        cnn_label = s[cnn_inds[t_ind]]
                        if len(cnn_statuses[player][output_key]) == 0:
                            cnn_statuses[player][output_key].append({'begin': 0, 'end': 0, 'status': cnn_label})
                        else:
                            if cnn_label == cnn_statuses[player][output_key][-1]['status']:
                                cnn_statuses[player][output_key][-1]['end'] = current_time
                            else:
                                cnn_statuses[player][output_key].append(
                                    {'begin': current_time, 'end': current_time, 'status': cnn_label})
            cur_kf = [{'time_point': round(time + (t_ind * time_step), 1), 'slots': {}} for t_ind in
                      range(frames_per_seq)]

            slot_inputs = [np.zeros((frames_per_seq, 50))[None]]

            for slot in lstm_kfs.keys():
                intermediate_output = kf_cnn_intermediate_model.predict(to_predicts_kf[slot])
                slot_inputs.append(intermediate_output[None])
            lstm_output = kf_lstm_model.predict(slot_inputs)

            for t_ind in range(frames_per_seq):
                for slot in range(6):
                    d = {}
                    for output_ind, (output_key, s) in enumerate(kf_sets.items()):

                        lstm_inds = lstm_output[slot * 6 + output_ind].argmax(axis=2)
                        lstm_label = s[lstm_inds[0, t_ind]]
                        d[output_key] = lstm_label
                    if d['second_hero'] != 'n/a':
                        cur_kf[t_ind]['slots'][slot] = d
            kf.extend(cur_kf)

            time += (j * time_step)
            to_predicts = {k: np.zeros(shape, dtype=np.uint8) for k in lstm_statuses.keys()}
            j = 0
    if j != 0:
        for player in lstm_statuses.keys():
            intermediate_output = player_cnn_intermediate_model.predict(to_predicts[player])
            lstm_output = player_lstm_model.predict(intermediate_output[None])
            cnn_output = player_cnn_model.predict(to_predicts[player])
            for output_ind, (output_key, s) in enumerate(player_sets.items()):
                cnn_inds = cnn_output[output_ind].argmax(axis=1)
                label_inds = lstm_output[output_ind].argmax(axis=2)
                for t_ind in range(label_inds.shape[0]):
                    current_time = time + (t_ind * time_step)
                    lstm_label = s[label_inds[0, t_ind]]
                    if len(lstm_statuses[player][output_key]) == 0:
                        lstm_statuses[player][output_key].append({'begin': 0, 'end': 0, 'status': lstm_label})
                    else:
                        if lstm_label == lstm_statuses[player][output_key][-1]['status']:
                            lstm_statuses[player][output_key][-1]['end'] = current_time
                        else:
                            lstm_statuses[player][output_key].append(
                                {'begin': current_time, 'end': current_time, 'status': lstm_label})
                    cnn_label = s[cnn_inds[t_ind]]
                    if len(cnn_statuses[player][output_key]) == 0:
                        cnn_statuses[player][output_key].append({'begin': 0, 'end': 0, 'status': cnn_label})
                    else:
                        if cnn_label == cnn_statuses[player][output_key][-1]['status']:
                            cnn_statuses[player][output_key][-1]['end'] = current_time
                        else:
                            cnn_statuses[player][output_key].append(
                                {'begin': current_time, 'end': current_time, 'status': cnn_label})
    # for k, v in cnn_statuses.items():
    #    print(k)
    #    for k2, v2 in v.items():
    #        print(k2)
    #        print(v2)
    # print('cnn', generate_events(cnn_statuses))
    # for k, v in lstm_statuses.items():
    #    print(k)
    #    for k2, v2 in v.items():
    #        print(k2)
    #        print(v2)
    player_states = generate_states(lstm_statuses)
    print('lstm', )
    generate_kill_events(kf, player_states)


def generate_kill_events(kf, player_states):
    death_events = []
    for p, s in player_states.items():
        death_events.extend(s.generate_death_events())
    print(sorted(death_events))
    print('KILL FEED')
    possible_events = []
    for ind, k in enumerate(kf):
        for slot in range(6):
            if slot not in k['slots']:
                continue
            prev_events = []
            if ind != 0:
                if 0 in kf[ind-1]['slots']:
                    prev_events.append(kf[ind-1]['slots'][0])
                for j in range(slot,0, -1):
                    if j in kf[ind-1]['slots']:
                        prev_events.append(kf[ind-1]['slots'][j])
            e = k['slots'][slot]
            if e in prev_events:
                for p_ind, poss_e in enumerate(possible_events):
                    if e == poss_e['event'] and poss_e['time_point'] + poss_e['duration'] + 0.15 >= k['time_point']:
                        possible_events[p_ind]['duration'] = k['time_point'] - poss_e['time_point']
            else:
                possible_events.append({'time_point': k['time_point'], 'duration':0, 'event':e})


    for i, p in enumerate(possible_events):
        print(p)


class PlayerState(object):
    def __init__(self, player, statuses):
        self.player = player
        self.statuses = statuses
        self.color = self.statuses['color'][0]['status']

    def hero_at_time(self, time_point):
        for hero_state in self.statuses['hero']:
            if hero_state['end'] >= time_point >= hero_state['begin']:
                return hero_state['status']

    def generate_death_events(self):
        deaths = []
        print(self.player)
        for alive_state in self.statuses['alive']:
            print(alive_state)
            if alive_state['status'] == 'dead':
                deaths.append([alive_state['begin'] + 0.2, (self.hero_at_time(alive_state['begin']), self.color)])
        return deaths

    def generate_switches(self):
        switches = []
        for hero_state in self.statuses['hero']:
            switches.append([self.player, hero_state['begin'], hero_state['status']])

    def generate_ults(self):
        ult_gains, ult_uses = [], []
        for i, ult_state in enumerate(self.statuses['has_ult']):

            if ult_state['status'] == 'has_ult':
                ult_gains.append([self.player, ult_state['begin']])
            elif i > 0 and ult_state['status'] == 'no_ult':
                ult_uses.append([self.player, ult_state['begin']])
        return ult_gains, ult_uses


def generate_states(statuses):
    player_states = {}
    for player, s in statuses.items():
        player_states[player] = PlayerState(player, s)
    return player_states


def load_player_cnn_model():
    final_output_weights = os.path.join(working_dir, 'player_status_cnn', 'player_weights.h5')
    final_output_json = os.path.join(working_dir, 'player_status_cnn', 'player_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    embedding_model = keras.models.Model(inputs=model.input,
                                         outputs=model.get_layer('representation').output)
    return model, embedding_model


def load_player_lstm_model():
    final_output_weights = os.path.join(working_dir, 'player_status_lstm', 'player_weights.h5')
    final_output_json = os.path.join(working_dir, 'player_status_lstm', 'player_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    return model


def load_kf_cnn_model():
    final_output_weights = os.path.join(working_dir, 'kf_cnn', 'kf_weights.h5')
    final_output_json = os.path.join(working_dir, 'kf_cnn', 'kf_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    embedding_model = keras.models.Model(inputs=model.input,
                                         outputs=model.get_layer('representation').output)
    return model, embedding_model


def load_kf_lstm_model():
    final_output_weights = os.path.join(working_dir, 'kf_lstm', 'kf_weights.h5')
    final_output_json = os.path.join(working_dir, 'kf_lstm', 'kf_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    return model


if __name__ == '__main__':
    kf_cnn_model, kf_cnn_intermediate_model = load_kf_cnn_model()
    kf_lstm_model = load_kf_lstm_model()
    player_cnn_model, player_cnn_intermediate_model = load_player_cnn_model()
    player_lstm_model = load_player_lstm_model()

    print('loaded model')
    to_annotate = r"E:\Data\Overwatch\raw_data\annotations\matches\2372\1\1.mp4"
    begin, end = 488 + 300, 488 + 360
    # to_annotate = r"E:\Data\Overwatch\raw_data\annotations\matches\2360\2360.mp4"
    # begin, end = 2000, 2100
    # to_annotate = r"E:\Data\Overwatch\raw_data\annotations\matches\2374\3\3.mp4"
    # begin, end = 438 + 60, 438 + 120
    predict_on_video(to_annotate, begin, end)
