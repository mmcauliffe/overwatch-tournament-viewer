import os
import json
import cv2
import numpy as np

working_dir = r'E:\Data\Overwatch\models'

kf_train_dir = r'E:\Data\Overwatch\training_data\kf_cnn'
player_train_dir = r'E:\Data\Overwatch\training_data\player_status_cnn'
mid_train_dir = r'E:\Data\Overwatch\training_data\mid_cnn'

time_step = 0.1
frames_per_seq = 100

from annotator.utils import get_annotate_rounds, get_local_path, get_local_file, update_annotations, BOX_PARAMETERS, \
    HERO_SET, ABILITY_SET, COLOR_SET, FileVideoStream, Empty

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
kf_label_file = os.path.join(kf_train_dir, 'labels.txt')

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

kf_labels = load_set(kf_label_file)

# kf_sets = {}
# for k, v in kf_set_files.items():
#    kf_sets[k] = load_set(v)

mid_sets = {}


# for k, v in mid_set_files.items():
#    mid_sets[k] = load_set(v)


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
    shape = (6, kf_params['WIDTH'], kf_params['HEIGHT'], 3)
    to_predict = np.zeros(shape, dtype=np.uint8)

    for i in range(6):
        y = kf_params['Y']
        y += (kf_params['HEIGHT'] + kf_params['MARGIN']) * i
        box = frame[y: y + kf_params['HEIGHT'],
              kf_params['X']: kf_params['X'] + kf_params['WIDTH']]
        to_predict[i, ...] = np.swapaxes(box, 1, 0)[None]
    output = kf_model.predict(to_predict)

    output = convert_kf_output(output)
    for f in range(6):
        slot = {}
        for k, v in output_dict.items():
            slot[k] = v[f]
        if all(x == 'n/a' for x in slot.values()):
            continue
        kf.append(list(slot.values()))



def annotate_statuses(to_predict, statuses, beg_time):
    for player in statuses.keys():
        lstm_output = player_cnn_model.predict(to_predict[player][None])
        for output_ind, (output_key, s) in enumerate(player_sets.items()):
            label_inds = lstm_output[output_ind].argmax(axis=2)
            for t_ind in range(frames_per_seq):
                current_time = beg_time + (t_ind * time_step)
                lstm_label = s[label_inds[0, t_ind]]
                if lstm_label != 'n/a':
                    if len(statuses[player][output_key]) == 0:
                        statuses[player][output_key].append({'begin': 0, 'end': 0, 'status': lstm_label})
                    else:
                        if lstm_label == statuses[player][output_key][-1]['status']:
                            statuses[player][output_key][-1]['end'] = current_time
                        elif statuses[player][output_key][-1]['end'] - statuses[player][output_key][-1][
                            'begin'] < 0.5:
                            if len(statuses[player][output_key]) > 2 and lstm_label == \
                                    statuses[player][output_key][-2]['status']:
                                del statuses[player][output_key][-1]
                                statuses[player][output_key][-1]['end'] = current_time

                        else:
                            statuses[player][output_key].append(
                                {'begin': current_time, 'end': current_time, 'status': lstm_label})
    return statuses


from collections import defaultdict

def convert_kf_output(output):
    intervals = []
    for i in range(output.shape[0]):
        lab = kf_labels[output[i]]
        if not intervals or lab != intervals[-1]['label']:
            intervals.append({'begin': i, 'end': i, 'label': lab})
        else:
            intervals[-1]['end'] = i
    intervals = [x for x in intervals if x['begin'] != x['end']]
    data = {'first_hero': kf_labels[0],
            'first_color': kf_labels[0],
            'ability': kf_labels[0],
            'headshot': kf_labels[0],
            'second_hero': kf_labels[0],
            'second_color': kf_labels[0]}
    if len(intervals) == 1 and intervals[0]['label'] == kf_labels[0]:
        return data
    first_intervals = []
    second_intervals = []
    ability_intervals = []
    for i in intervals:
        if i['label'] in ABILITY_SET:
            ability_intervals.append(i)
        if not len(ability_intervals):
            first_intervals.append(i)
        elif i['label'] not in ABILITY_SET:
            second_intervals.append(i)

    color_counts = defaultdict(int)
    hero_counts = defaultdict(int)
    for i in first_intervals:
        if i['label'] in COLOR_SET:
            color_counts[i['label']] += i['end'] - i['begin']
        elif i['label'] in HERO_SET:
            hero_counts[i['label']] += i['end'] - i['begin']

    if color_counts:
        data['first_color'] = max(color_counts.keys(), key=lambda x: color_counts[x])
    if hero_counts:
        data['first_hero'] = max(hero_counts.keys(), key=lambda x: hero_counts[x])
    ability_counts = defaultdict(int)
    for i in ability_intervals:
        ability_counts[i['label']] += i['end'] - i['begin']
    if ability_counts:
        data['ability'] = max(ability_counts.keys(), key=lambda x: ability_counts[x])
        if data['ability'].endswith('_headshot'):
            data['headshot'] = True
            data['ability'] = data['ability'].split('_')[0]
        else:
            data['headshot'] = False

    color_counts = defaultdict(int)
    hero_counts = defaultdict(int)
    for i in second_intervals:
        if i['label'] in COLOR_SET:
            color_counts[i['label']] += i['end'] - i['begin']
        elif i['label'] in HERO_SET:
            hero_counts[i['label']] += i['end'] - i['begin']

    if color_counts:
        data['second_color'] = max(color_counts.keys(), key=lambda x: color_counts[x])
    if hero_counts:
        data['second_hero'] = max(hero_counts.keys(), key=lambda x: hero_counts[x])

    return data

def annotate_kf(to_predict):
    cur_kf = {}
    output = kf_cnn_model.predict(to_predict)
    for slot in range(6):
        cnn_inds = output[slot].argmax(axis=1)
        s = convert_kf_output(cnn_inds)
        if s['second_hero'] != 'n/a':
            cur_kf[slot] = s

    return cur_kf


def predict_on_video(video_path, begin, end):
    from datetime import timedelta
    print('beginning prediction')
    frames = (end - begin) / time_step
    num_players = 1
    fvs = FileVideoStream(video_path, begin, end, time_step).start()
    left_params = BOX_PARAMETERS['REGULAR']['LEFT']
    right_params = BOX_PARAMETERS['REGULAR']['RIGHT']
    kf_params = BOX_PARAMETERS['REGULAR']['KILL_FEED_SLOT']
    shape = (frames_per_seq, left_params['HEIGHT'], left_params['WIDTH'], 3)
    kf_shape = (6,kf_params['WIDTH'], kf_params['HEIGHT'],  3)
    j = 0
    lstm_statuses = {}
    cnn_statuses = {}
    for side in ['left', 'right']:
        for i in range(6):
            lstm_statuses[(side, i)] = {k: [] for k in player_sets.keys()}
            cnn_statuses[(side, i)] = {k: [] for k in player_sets.keys()}
    to_predicts = {k: np.zeros(shape, dtype=np.uint8) for k in lstm_statuses.keys()}
    expected_sequence_count = int(frames / frames_per_seq) + 1
    time = 0
    seq_ind = 0
    print(begin, end, end - begin)
    frame_ind = 0
    kf = []
    while True:
        try:
            frame, time_point = fvs.read()
        except Empty:
            break
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
        to_predicts_kf = np.zeros(kf_shape, dtype=np.uint8)
        for slot in range(6):
            x = kf_params['X']
            y = kf_params['Y']
            y += (kf_params['HEIGHT'] + kf_params['MARGIN']) * (slot)
            box = frame[y: y + kf_params['HEIGHT'],
                  x: x + kf_params['WIDTH']]
            to_predicts_kf[slot, ...] = np.swapaxes(box, 1, 0)[None]
        cur_kf = annotate_kf(to_predicts_kf)
        kf.append({'time_point': time_point, 'slots': cur_kf})
        j += 1
        if j == frames_per_seq:
            print('sequences', seq_ind, expected_sequence_count)
            seq_ind += 1
            lstm_statuses = annotate_statuses(to_predicts, lstm_statuses, time)


            time += (j * time_step)
            to_predicts = {k: np.zeros(shape, dtype=np.uint8) for k in lstm_statuses.keys()}
            j = 0
    if j != 0:
        lstm_statuses = annotate_statuses(to_predicts, lstm_statuses, time)


    player_states = generate_states(lstm_statuses)
    print('lstm', )
    generate_kill_events(kf, player_states)
    for k, v in player_states.items():
        print(k)
        print('switches', v.generate_switches())
        ug, uu = v.generate_ults()
        print('ult_gains', ug)
        print('ult_uses', uu)
    return player_states


def close_events(e_one, e_two):
    if e_one['first_hero'] != e_two['first_hero']:
        return False
    if e_one['second_hero'] != e_two['second_hero']:
        return False
    if e_one['first_color'] != e_two['first_color']:
        return False
    if e_one['second_color'] != e_two['second_color']:
        return False
    #if e_one['ability'] != e_two['ability']:
    #    return False
    return True

def merged_event(e_one, e_two):
    if e_one['event'] == e_two['event']:
        return e_one['event']
    elif e_one['duration'] > e_two['duration']:
        return e_one['event']
    return e_two['event']


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
                if 0 in kf[ind - 1]['slots']:
                    prev_events.append(kf[ind - 1]['slots'][0])
                for j in range(slot, 0, -1):
                    if j in kf[ind - 1]['slots']:
                        prev_events.append(kf[ind - 1]['slots'][j])
            e = k['slots'][slot]
            if e in prev_events:
                for p_ind, poss_e in enumerate(possible_events):
                    if e == poss_e['event'] and poss_e['time_point'] + poss_e['duration'] + 0.15 >= k['time_point']:
                        possible_events[p_ind]['duration'] = k['time_point'] - poss_e['time_point']
            else:
                possible_events.append({'time_point': k['time_point'], 'duration': 0, 'event': e})
    better_possible_events = []
    for i, p in enumerate(possible_events):
        for j, p2 in enumerate(better_possible_events):
            p2_end = p2['time_point'] + p2['duration']
            if close_events(p['event'], p2['event']) and abs(p2_end - p['time_point']) < 0.25:
                better_possible_events[j]['duration'] += p['duration']
                better_possible_events[j]['event'] = merged_event(p, p2)
                break
        else:
            better_possible_events.append(p)


    for p in better_possible_events:
        if not p['duration']:
            continue
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
        return switches

    def generate_ults(self):
        ult_gains, ult_uses = [], []
        for i, ult_state in enumerate(self.statuses['ult']):

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
    # embedding_model = keras.models.Model(inputs=model.input,
    #                                     outputs=model.get_layer('representation').output)
    return model  # , embedding_model


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
    return model


def load_kf_lstm_model():
    final_output_weights = os.path.join(working_dir, 'kf_lstm', 'kf_weights.h5')
    final_output_json = os.path.join(working_dir, 'kf_lstm', 'kf_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    return model


def main():
    rounds = get_annotate_rounds()
    for r in rounds:
        local_path = get_local_path(r)
        if local_path is None:
            print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
            get_local_file(r)
    for r in rounds:
        print(r)
        data = predict_on_video(get_local_path(r), r['begin'], r['end'])
        print(data)
        update_annotations(data, r['id'])
        error


if __name__ == '__main__':
    import keras

    kf_cnn_model = load_kf_cnn_model()
    # kf_lstm_model = load_kf_lstm_model()
    player_cnn_model = load_player_cnn_model()
    # player_lstm_model = load_player_lstm_model()

    print('loaded model')
    main()
