import os
import json
import cv2
import numpy as np
import re

working_dir = r'E:\Data\Overwatch\models'

kf_train_dir = r'E:\Data\Overwatch\training_data\kf_cnn_slot'
replay_train_dir = r'E:\Data\Overwatch\training_data\replay_cnn'
player_train_dir = r'E:\Data\Overwatch\training_data\player_status_cnn'
mid_train_dir = r'E:\Data\Overwatch\training_data\mid_cnn'
annotation_dir = r'E:\Data\Overwatch\annotations'
oi_annotation_dir = r'E:\Data\Overwatch\oi_annotations'

time_step = 0.1
frames_per_seq = 100

from annotator.utils import get_annotate_rounds, get_local_path, get_local_file, update_annotations, BOX_PARAMETERS, \
    HERO_SET, ABILITY_SET, COLOR_SET, FileVideoStream, Empty, get_annotate_vods, get_vod_path, get_local_vod, upload_game

player_set_files = {
    'hero': os.path.join(player_train_dir, 'hero_set.txt'),
    'color': os.path.join(player_train_dir, 'color_set.txt'),
    'alive': os.path.join(player_train_dir, 'alive_set.txt'),
    'ult': os.path.join(player_train_dir, 'ult_set.txt'),
}
kf_set_files = {
    'first_hero': os.path.join(kf_train_dir, 'first_hero_set.txt'),
    'first_color': os.path.join(kf_train_dir, 'first_color_set.txt'),
    'ability': os.path.join(kf_train_dir, 'ability_set.txt'),
    'second_hero': os.path.join(kf_train_dir, 'second_hero_set.txt'),
    'second_color': os.path.join(kf_train_dir, 'second_color_set.txt'),

}
kf_label_file = os.path.join(kf_train_dir, 'labels.txt')
replay_set_file = os.path.join(replay_train_dir, 'replay_set.txt')
spectator_mode_file = os.path.join(replay_train_dir, 'spectator_mode_set.txt')

mid_set_files = {
                 'overtime': os.path.join(mid_train_dir, 'overtime_set.txt'),
                 'point_status': os.path.join(mid_train_dir, 'point_set.txt'),
                 }

mid_end_set_files = {
                 'attacking_color': os.path.join(mid_train_dir, 'color_set.txt'),
                 'map': os.path.join(mid_train_dir, 'map_set.txt'),
                 'map_mode': os.path.join(mid_train_dir, 'map_mode_set.txt'),
                 'round_number': os.path.join(mid_train_dir, 'round_number_set.txt'),
                 'spectator_mode': os.path.join(mid_train_dir, 'spectator_mode_set.txt'),
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
replay_set = load_set(replay_set_file)
spectator_modes = load_set(spectator_mode_file)
spectator_mode_count = len(spectator_modes)

kf_sets = {}
for k, v in kf_set_files.items():
    kf_sets[k] = load_set(v)

mid_sets = {}

for k, v in mid_set_files.items():
    mid_sets[k] = load_set(v)
mid_end_sets = {}

for k, v in mid_end_set_files.items():
    mid_end_sets[k] = load_set(v)


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



def annotate_statuses(to_predict, statuses, beg_time, spectator_mode):
    spectator_mode_input = np.zeros((to_predict[('left', 0)].shape[0], 100, spectator_mode_count))
    m = sparsify(np.array([spectator_modes.index(spectator_mode)]), spectator_mode_count)
    for i in range(to_predict[('left', 0)].shape[0]):
        spectator_mode_input[i, :] = m
    for player in statuses.keys():
        lstm_output = player_cnn_model.predict([to_predict[player][None], spectator_mode_input])
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
                        #elif statuses[player][output_key][-1]['end'] - statuses[player][output_key][-1][
                        #    'begin'] < 0.5:
                        #    if len(statuses[player][output_key]) > 2 and lstm_label == \
                        #            statuses[player][output_key][-2]['status']:
                        #        del statuses[player][output_key][-1]
                        #        statuses[player][output_key][-1]['end'] = current_time

                        else:
                            statuses[player][output_key].append(
                                {'begin': current_time, 'end': current_time, 'status': lstm_label})
    return statuses


def annotate_mid(to_predict,statuses, beg_time):

    lstm_output = mid_model.predict([to_predict[None]])
    for output_ind, (output_key, s) in enumerate(list(mid_sets.items()) + list(mid_end_sets.items())):
        if output_key in mid_end_sets:
            label_inds = lstm_output[output_ind].argmax(axis=1)
            lstm_label = s[label_inds[0]]
            if lstm_label != 'n/a':
                if len(statuses[output_key]) == 0:
                    statuses[output_key].append({'begin': beg_time, 'end': beg_time+ time_step*frames_per_seq, 'status': lstm_label})
                else:
                    if lstm_label == statuses[output_key][-1]['status']:
                        statuses[output_key][-1]['end'] = beg_time+ time_step*frames_per_seq
                    #elif statuses[player][output_key][-1]['end'] - statuses[player][output_key][-1][
                    #    'begin'] < 0.5:
                    #    if len(statuses[player][output_key]) > 2 and lstm_label == \
                    #            statuses[player][output_key][-2]['status']:
                    #        del statuses[player][output_key][-1]
                    #        statuses[player][output_key][-1]['end'] = current_time

                    else:
                        statuses[output_key].append(
                            {'begin': beg_time, 'end': beg_time+ time_step*frames_per_seq, 'status': lstm_label})
        else:
            label_inds = lstm_output[output_ind].argmax(axis=2)
            for t_ind in range(frames_per_seq):
                current_time = beg_time + (t_ind * time_step)
                lstm_label = s[label_inds[0, t_ind]]
                if lstm_label != 'n/a':
                    if len(statuses[output_key]) == 0:
                        statuses[output_key].append({'begin': 0, 'end': 0, 'status': lstm_label})
                    else:
                        if lstm_label == statuses[output_key][-1]['status']:
                            statuses[output_key][-1]['end'] = current_time
                        #elif statuses[player][output_key][-1]['end'] - statuses[player][output_key][-1][
                        #    'begin'] < 0.5:
                        #    if len(statuses[player][output_key]) > 2 and lstm_label == \
                        #            statuses[player][output_key][-2]['status']:
                        #        del statuses[player][output_key][-1]
                        #        statuses[player][output_key][-1]['end'] = current_time

                        else:
                            statuses[output_key].append(
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


    #print('first_intervals', first_intervals)
    #print('ability_intervals', ability_intervals)
    #print('second_intervals', second_intervals)
    return data

def sparsify(y, n_classes):
    'Returns labels in binary NumPy array'
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])

def annotate_kf(to_predict, spectator_mode):
    spectator_mode_input = np.zeros((to_predict.shape[0], 62, spectator_mode_count))
    m = sparsify(np.array([spectator_modes.index(spectator_mode)]), spectator_mode_count)
    for i in range(to_predict.shape[0]):
        spectator_mode_input[i, :] = m
    cur_kf = {}

    output = kf_cnn_model.predict([to_predict,spectator_mode_input])
    for slot in range(6):
        #print(slot, output[slot])
        cnn_inds= output[slot].argmax(axis=1)
        #print(cnn_inds)
        s = convert_kf_output(cnn_inds)
        if s['second_hero'] != 'n/a':
            #print(s)
            #cv2.imshow('frame', np.swapaxes(to_predict[slot, ...], 0, 1))
            cur_kf[slot] = s
            #cv2.waitKey(0)
        #d = {}
        #for output_ind, (output_key, s) in enumerate(kf_sets.items()):
        #    d[output_key] = s[cnn_inds]
        #    cnn_inds = output[output_ind][slot].argmax(axis=0)
        #if d['first_color'] not in ['n/a', 'white']:
        #    d['first_color'] = 'nonwhite'
        #if d['second_color'] not in ['n/a', 'white']:
        #    d['second_color'] = 'nonwhite'
        #if d['second_hero'] != 'n/a':
        #    cur_kf[slot] = d
    return cur_kf


def predict_on_video(video_path, begin, end, spectator_mode, film_format, sequences):
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
    from datetime import timedelta
    debug = False
    print('beginning prediction')
    frames = (end - begin) / time_step
    num_players = 1
    left_params = BOX_PARAMETERS[film_format]['LEFT']
    right_params = BOX_PARAMETERS[film_format]['RIGHT']
    kf_params = BOX_PARAMETERS[film_format]['KILL_FEED_SLOT']

    mid_params = BOX_PARAMETERS[film_format]['MID']
    shape = (frames_per_seq, left_params['HEIGHT'], left_params['WIDTH'], 3)
    kf_shape = (6,kf_params['WIDTH'], kf_params['HEIGHT'],  3)
    mid_shape = (frames_per_seq, int(mid_params['HEIGHT'] * 0.3), int(mid_params['WIDTH'] * 0.3), 3)
    expected_sequence_count = int(frames / frames_per_seq) + 1
    seq_ind = 0
    print(begin, end, end - begin)
    kf = []
    lstm_statuses = {}
    labels = {}
    for side in ['left', 'right']:
        for i in range(6):
            lstm_statuses[(side, i)] = {k: [] for k in player_sets.keys()}
            labels[(side, i)] = defaultdict(int)
    for s in sequences:
        print(s)
        time = s['begin']
        fvs = FileVideoStream(video_path, s['begin']+begin, s['end']+ begin, time_step, real_begin=begin).start()
        j = 0
        to_predicts = {k: np.zeros(shape, dtype=np.uint8) for k in lstm_statuses.keys()}
        to_predict_mid = np.zeros(mid_shape, dtype=np.uint8)
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            #if time_point < 108:
            #    continue
            #print(time_point)
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
                name_box = box[34:46, :]
                gray = cv2.cvtColor(name_box, cv2.COLOR_BGR2GRAY)
                input = np.expand_dims(cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1], -1)
                input = np.swapaxes(input, 0, 1)[None]
                label = ocr_model(input)[0]
                labels[player][''.join(label)] += 1
                if debug:
                    cv2.imshow('player_frame_{}_{}'.format(*player), box)
                    cv2.imshow('name_frame_{}_{}'.format(*player), gray)
                to_predicts[player][j, ...] = box[None]

            box = frame[mid_params['Y']: mid_params['Y'] + mid_params['HEIGHT'],
                  mid_params['X']: mid_params['X'] + mid_params['WIDTH']]
            box = cv2.resize(box, (0, 0), fx=0.3, fy=0.3)
            to_predict_mid[j, ...] = box[None]
            to_predicts_kf = np.zeros(kf_shape, dtype=np.uint8)
            for slot in range(6):
                x = kf_params['X']
                y = kf_params['Y']
                y += (kf_params['HEIGHT'] + kf_params['MARGIN']) * (slot)
                box = frame[y: y + kf_params['HEIGHT'],
                      x: x + kf_params['WIDTH']]
                if debug:
                    cv2.imshow('kf_slot_{}'.format(slot), box)
                to_predicts_kf[slot, ...] = np.swapaxes(box, 1, 0)[None]
                #cv2.imshow('frame_{}'.format(slot), box)
            cur_kf = annotate_kf(to_predicts_kf, spectator_mode)
            #print(cur_kf)
            #cv2.waitKey(0)
            kf.append({'time_point': time_point, 'slots': cur_kf})
            if debug:
                cv2.waitKey(0)
            j += 1
            if j == frames_per_seq:
                print('sequences', seq_ind, expected_sequence_count, time)
                seq_ind += 1
                lstm_statuses = annotate_statuses(to_predicts, lstm_statuses, time, spectator_mode)
                time += (j * time_step)
                to_predicts = {k: np.zeros(shape, dtype=np.uint8) for k in lstm_statuses.keys()}
                j = 0
        if j != 0:
            lstm_statuses = annotate_statuses(to_predicts, lstm_statuses, time, spectator_mode)
    #print(mid_status)
    #error
    for k, v in lstm_statuses.items():
        print(k)
        for k2, v2 in v.items():
            print(k2)
            print(v2)
            new_series = []
            for interval in v2:
                if interval['end'] - interval['begin'] > 0.3:
                    if len(new_series) > 0 and new_series[-1]['status'] == interval['status']:
                        new_series[-1]['end'] = interval['end']
                    else:
                        if len(new_series) > 0 and interval['begin'] != new_series[-1]['end']:
                            interval['begin'] = new_series[-1]['end']
                        if k2 == 'hero':
                            if interval['begin'] != 0:
                                check = False
                                for s in v['alive']:
                                    #print(s)
                                    if s['begin'] <= interval['begin'] < s['end'] and interval['end']- interval['begin'] < 10:
                                        check = True
                                        break

                                #print(s, interval)
                                if check and s['status'] == 'dead':
                                    continue
                                #for s in v['ult']:
                                #    print(s)
                                #    if s['begin'] <= interval['begin'] < s['end']:
                                #        break
                                #print(s, interval)
                                #if s['status'] == 'has_ult':
                                #    continue
                        new_series.append(interval)

            lstm_statuses[k][k2] = new_series
    #print(k, lstm_statuses[k]['hero'])
    player_states = generate_states(lstm_statuses)
    for p, s in player_states.items():
        side = p[0]
        if side == 'left':
            left_color = s.color
        else:
            right_color = s.color
    #print('lstm', )
    kill_feed_events = generate_kill_events(kf, player_states)
    #for k, v in player_states.items():
    #    print(k)
    #    print('switches', v.generate_switches())
    #    ug, uu = v.generate_ults()
    #    print('ult_gains', ug)
    #    print('ult_uses', uu)
    data_player_states = {}
    for k, v in player_states.items():
        k = '{}_{}'.format(*k)
        data_player_states[k] = {}
        switches = v.generate_switches()
        ug, uu = v.generate_ults()

        data_player_states[k]['switches'] = switches
        data_player_states[k]['ult_gains'] = ug
        data_player_states[k]['ult_uses'] = uu
    for k, v in labels.items():
        k = '{}_{}'.format(*k)

        data_player_states[k]['player_name'] = max(v.keys(), key=lambda x:v[x])
    return {'player':data_player_states, 'kill_feed':kill_feed_events, 'left_color': left_color, 'right_color':right_color}


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
    #death_events = []
    #for p, s in player_states.items():
    #    death_events.extend(s.generate_death_events())
    #print(death_events)
    #print(sorted(death_events))
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
            if close_events(p['event'], p2['event']) and abs(p2_end - p['time_point']) <= 1.5 and p2['duration'] + p['duration'] < 8:
                better_possible_events[j]['duration'] += p['duration']
                better_possible_events[j]['event'] = merged_event(p, p2)
                break
            elif close_events(p['event'], p2['event']) and p2_end > p['time_point'] > p2['time_point']:
                break

        else:
            better_possible_events.append(p)
    better_possible_events = [x for x in better_possible_events if x['duration'] > 1]
    return better_possible_events


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
            switches.append([hero_state['begin'], hero_state['status']])
        return switches

    def generate_ults(self):
        ult_gains, ult_uses = [], []
        for i, ult_state in enumerate(self.statuses['ult']):

            if ult_state['status'] == 'has_ult':
                ult_gains.append([ult_state['begin']])
            elif i > 0 and ult_state['status'] == 'no_ult':
                ult_uses.append([ult_state['begin']])
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

def load_player_ocr_model():
    from functools import partial
    characters = load_set(r'E:\Data\Overwatch\training_data\player_status_cnn\characters.txt')
    def labels_to_text(ls):
        ret = []
        for c in ls:
            if c >= len(characters):
                continue
            ret.append(characters[c])
        return ret
    import itertools
    def decode_batch(test_func, word_batch):
        out = test_func([word_batch])
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = labels_to_text(out_best)
            ret.append(outstr)
        return ret
    final_output_weights = os.path.join(working_dir, 'player_ocr_ctc', 'ocr_weights.h5')
    final_output_json = os.path.join(working_dir, 'player_ocr_ctc', 'ocr_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    embedding_model = keras.models.Model(inputs=[model.input[0]],
                                         outputs=[model.get_layer('softmax').output])
    return partial(decode_batch, embedding_model.predict_on_batch)



def load_player_lstm_model():
    final_output_weights = os.path.join(working_dir, 'player_status_lstm', 'player_weights.h5')
    final_output_json = os.path.join(working_dir, 'player_status_lstm', 'player_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    return model


def load_kf_cnn_model():
    final_output_weights = os.path.join(working_dir, 'kf_cnn_slot', 'kf_weights.h5')
    final_output_json = os.path.join(working_dir, 'kf_cnn_slot', 'kf_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    return model


def load_game_model():
    final_output_weights = os.path.join(working_dir, 'game_cnn', 'game_weights.h5')
    final_output_json = os.path.join(working_dir, 'game_cnn', 'game_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    return model

def load_replay_model():
    final_output_weights = os.path.join(working_dir, 'replay_cnn', 'replay_weights.h5')
    final_output_json = os.path.join(working_dir, 'replay_cnn', 'replay_model.json')
    with open(final_output_json, 'r') as f:
        loaded_model_json = f.read()
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(final_output_weights)
    return model

def load_mid_model():
    final_output_weights = os.path.join(working_dir, 'mid_cnn', 'mid_weights.h5')
    final_output_json = os.path.join(working_dir, 'mid_cnn', 'mid_model.json')
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


def get_replays(video_path, begin, end, film_format):
    replay_time_step = 5
    debug = False
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
    print('beginning prediction')
    time_step = 0.1
    frames = (end - begin) / time_step
    num_players = 1
    fvs = FileVideoStream(video_path, begin, end, replay_time_step).start()
    replay_params = BOX_PARAMETERS[film_format]['REPLAY']
    pause_params = BOX_PARAMETERS[film_format]['PAUSE']

    print(begin, end, end - begin)
    frame_ind = 0
    begin_margin = 2.5
    end_margin = 0.6
    replays_found = []
    pauses_found = []
    while True:
        try:
            frame, time_point = fvs.read()
        except Empty:
            break
        time_point = round(time_point, 1)

        print(time_point)
        frame_ind += 1
        box = frame[replay_params['Y']: replay_params['Y'] + replay_params['HEIGHT'],
              replay_params['X']:replay_params['X'] + replay_params['WIDTH']]
        label = pytesseract.image_to_string(box, config=tessdata_dir_config)
        if label.lower() == 'replay':
            replays_found.append(time_point)

        box = frame[pause_params['Y']: pause_params['Y'] + pause_params['HEIGHT'],
              pause_params['X']:pause_params['X'] + pause_params['WIDTH']]

        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([15,0,150])
        upper_orange = np.array([35,255,255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = 255 -mask
        continue
        if np.mean(mask) > 190: # FIXME

            pauses_found.append(time_point)


    print(replays_found)
    print(pauses_found)
    replays = []
    for t in replays_found:
        already_done = False
        for r in replays:
            if r['end']>= t >= r['begin']:
                already_done = True
                break
        if already_done:
            continue
        r_b = t -15 + begin
        r_e = t+ 15 + begin
        fvs = FileVideoStream(video_path, r_b, r_e, time_step, real_begin=begin).start()
        replay = {'begin': None}
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            time_point = round(time_point, 1)
            print(time_point)
            frame_ind += 1
            box = frame[replay_params['Y']: replay_params['Y'] + replay_params['HEIGHT'],
                  replay_params['X']:replay_params['X'] + replay_params['WIDTH']]
            label = pytesseract.image_to_string(box, config=tessdata_dir_config)
            if label.lower() == 'replay' and replay['begin'] is None:
                replay['begin'] = time_point
            if label.lower() == 'replay':
                replay['end'] = time_point
        replay['begin'] -= begin_margin
        replay['end'] += end_margin
        replays.append(replay)

    pauses = []
    for t in pauses_found:
        already_done = False
        for r in pauses:
            if r['end']>= t >= r['begin']:
                already_done = True
                break
        if already_done:
            continue
        r_b = t -5 + begin
        fvs = FileVideoStream(video_path, r_b, end, time_step, real_begin=begin).start()
        pause = {'begin': None}
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            time_point = round(time_point, 1)
            print(time_point)
            frame_ind += 1

            box = frame[pause_params['Y']: pause_params['Y'] + pause_params['HEIGHT'],
                  pause_params['X']:pause_params['X'] + pause_params['WIDTH']]

            hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
            lower_orange = np.array([15, 0, 150])
            upper_orange = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower_orange, upper_orange)
            mask = 255 - mask
            if np.mean(mask) > 150 and pause['begin'] is None:
                pause['begin'] = time_point
            if np.mean(mask) > 150:
                pause['end'] = time_point
            pause['begin'] -= 0.1
            pause['end'] += 0.1
        pauses.append(pause)

    externals = sorted(replays + pauses, key=lambda x:x['begin'])
    if not externals:
        sequences = [{'begin': begin-begin, 'end':end-begin}]
    else:
        sequences = []
        prev_time = 0
        for r in externals:
            sequences.append({'begin':prev_time, 'end': r['begin']})
            prev_time = r['end']
        sequences.append({'begin': prev_time, 'end': end-begin})
    return replays, pauses, sequences

def main():
    ignore_switches = [7393, 7394, 7395, 7396, 7397, 7398, 7399, 7400]
    rounds = get_annotate_rounds()
    for r in rounds:
        local_path = get_local_path(r)
        if local_path is None:
            print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
            get_local_file(r)
    for r in rounds:
        print(r)
        replays, pauses, sequences = get_replays(get_local_path(r), r['begin'], r['end'], r['game']['match']['film_format'])
        print('r', replays)
        print('p', pauses)
        print('s', sequences)
        data = predict_on_video(get_local_path(r), r['begin'], r['end'], r['spectator_mode'].lower(), r['game']['match']['film_format'], sequences)
        data['replays'] = replays
        data['pauses'] = pauses
        if r['id'] in ignore_switches:
            data['ignore_switches'] = True
        else:
            data['ignore_switches'] = False
        print(data)
        print(r)
        update_annotations(data, r['id'])
        #error

def annotate_game(to_predict, statuses, beg_time, game_model, time_step):
    labels = ['not_in_game', 'in_game']
    lstm_output = game_model.predict([to_predict[None]])
    print(lstm_output.shape)
    label_inds = lstm_output.argmax(axis=2)
    for t_ind in range(frames_per_seq):
        current_time = beg_time + (t_ind * time_step)
        lstm_label = labels[label_inds[0, t_ind]]
        if len(statuses) == 0:
            statuses.append({'begin': 0, 'end': 0, 'status': lstm_label})
        else:
            if lstm_label == statuses[-1]['status']:
                statuses[-1]['end'] = current_time
            #elif statuses[player][output_key][-1]['end'] - statuses[player][output_key][-1][
            #    'begin'] < 0.5:
            #    if len(statuses[player][output_key]) > 2 and lstm_label == \
            #            statuses[player][output_key][-2]['status']:
            #        del statuses[player][output_key][-1]
            #        statuses[player][output_key][-1]['end'] = current_time

            else:
                statuses.append(
                    {'begin': current_time, 'end': current_time, 'status': lstm_label})

    return statuses


def calc_end_props(status):
    out_props = {}
    for k in ['attacking_color', 'map', 'map_mode', 'round_number', 'spectator_mode']:
        counts = defaultdict(float)
        for r in status[k]:
            counts[r['status']] += r['end'] - r['begin']
        out_props[k] = max(counts, key=lambda x: counts[x])
    return out_props

def calc_overtime(status):
    actual_overtime = []
    for i, r in enumerate(status['overtime']):
        if r['status'] == 'not_overtime' and r['end'] - r['begin'] < 2:
            continue
        if len(actual_overtime) and r['status'] == actual_overtime[-1]['status']:
            actual_overtime[-1]['end'] = r['end']
        else:
            actual_overtime.append(r)
    return actual_overtime

def get_round_status(vod_path, begin, end, film_format, sequences):
    time_step = 0.1
    resize_factor = 0.3
    time = 0
    mid_params = BOX_PARAMETERS[film_format]['MID']
    mid_shape = (frames_per_seq, int(mid_params['HEIGHT'] * resize_factor), int(mid_params['WIDTH'] * resize_factor), 3)
    round_status = {k: [] for k in list(mid_sets.keys()) + list(mid_end_sets.keys())}
    for s in sequences:
        fvs = FileVideoStream(vod_path, s['begin'], s['end'], time_step, real_begin=begin).start()
        print(s['begin'], s['end'])
        to_predict_mid = np.zeros(mid_shape, dtype=np.uint8)
        j = 0
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            box = frame[mid_params['Y']: mid_params['Y'] + mid_params['HEIGHT'],
                  mid_params['X']: mid_params['X'] + mid_params['WIDTH']]
            box = cv2.resize(box, (0, 0), fx=resize_factor, fy=resize_factor)
            to_predict_mid[j, ...] = box[None]
            j += 1
            if j == frames_per_seq:
                round_status = annotate_mid(to_predict_mid, round_status, time)
                to_predict_mid = np.zeros(mid_shape, dtype=np.uint8)
                time += (j * time_step)
                j = 0
        if j > 0:
            round_status = annotate_mid(to_predict_mid, round_status, time)
    print('finished!')
    round_props = calc_end_props(round_status)
    overtimes = calc_overtime(round_status)
    round_props['begin'] = begin
    round_props['end'] = end
    round_props['overtime'] = overtimes
    return round_props

def extract_info(v):
    owl_mapping = {'DAL': 'Dallas Fuel',
                   'PHI': 'Philadelphia Fusion', 'SEO': 'Seoul Dynasty',
                   'LDN': 'London Spitfire', 'SFS': 'San Francisco Shock', 'HOU': 'Houston Outlaws',
                   'BOS': 'Boston Uprising', 'VAL': 'Los Angeles Valiant', 'GLA': 'Los Angeles Gladiators',
                   'FLA': 'Florida Mayhem', 'SHD': 'Shanghai Dragons', 'NYE': 'New York Excelsior'}
    channel = v['channel']['name']
    vod_type = ''
    if channel.lower() == 'overwatchcontenders':
        vod_type = 'game'
        pattern = r'([\w ]+) vs ([\w ]+) \| ([\w ]+) Game (\d) \| ([\w :]+) \| ([\w ]+)'
        m = re.match(pattern, v['title'])
        if m is not None:
            valid = True
            print(m.groups())
            team_one, team_two, desc, game_number, sub, main = m.groups()
            event = main + ' - ' + sub
        else:
            return False, None, None, None, None, None
    elif channel.lower() == 'overwatchleague':
        vod_type = 'game'
        pattern = r'Game (\d+) (\w+) @ (\w+) \| ([\w ]+)'
        m = re.match(pattern, v['title'])
        if m is not None:
            valid = True

            game_number, team_one, team_two, desc = m.groups()
            team_one = owl_mapping[team_one]
            team_two = owl_mapping[team_two]
        else:
            return False, None, None, None, None, None
    return valid, team_one, team_two, game_number, desc, event

def get_rounds(v):

    valid, team_one, team_two, game_number, desc, event = extract_info(v)
    if not valid:
        return
    annotation_path = os.path.join(oi_annotation_dir, '{}.json'.format(v['id']))
    if not os.path.exists(annotation_path):
        print(team_one, team_two, game_number, desc)
        game = {'vod_id': v['id'], 'team_one': team_one, 'team_two':team_two, 'game_number':game_number, 'description': desc, 'event': event, 'rounds': []}
        round_path = os.path.join(annotation_dir, 'predicted_rounds.txt')
        import time as timepackage
        game_model = load_game_model()
        time_step = 1
        frames_per_seq = 100
        time = 0
        fvs = FileVideoStream(get_vod_path(v), 0, 0, time_step).start()
        mid_params = BOX_PARAMETERS["O"]['MID']
        mid_shape = (frames_per_seq, int(mid_params['HEIGHT'] * 0.5), int(mid_params['WIDTH'] * 0.5), 3)
        to_predict_mid = np.zeros(mid_shape, dtype=np.uint8)
        j = 0
        game_status = []
        print('begin ingame/outofgame processing')
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            box = frame[mid_params['Y']: mid_params['Y'] + mid_params['HEIGHT'],
                  mid_params['X']: mid_params['X'] + mid_params['WIDTH']]
            box = cv2.resize(box, (0, 0), fx=0.5, fy=0.5)
            to_predict_mid[j, ...] = box[None]
            j += 1
            if j == frames_per_seq:
                print(time_point)
                game_status = annotate_game(to_predict_mid, game_status, time, game_model, time_step)
                to_predict_mid = np.zeros(mid_shape, dtype=np.uint8)
                time += (j * time_step)
                j = 0
        if j >0:
            game_status = annotate_game(to_predict_mid, game_status, time, game_model, time_step)
        rounds = []
        for g in game_status:
            if g['status'] == 'in_game':
                dur = g['end'] - g['begin']
                if dur < 45:
                    continue
                rounds.append({'vod_id': v['id'], 'begin': g['begin'], 'end': g['end'] })
        print('finished!')
        ignore_switches = []
        for r in rounds:
            replays, pauses, sequences = get_replays(get_vod_path(v), r['begin'], r['end'], v['film_format'])
            print('r', replays)
            print('p', pauses)
            print('s', sequences)
            round_props = get_round_status(get_vod_path(v), r['begin'], r['end'], v['film_format'], sequences)

            data = predict_on_video(get_vod_path(v), r['begin'], r['end'], round_props['spectator_mode'], v['film_format'], sequences)
            data['replays'] = replays
            data['pauses'] = pauses
            data['round'] = round_props
            if v['id'] in ignore_switches:
                data['ignore_switches'] = True
            else:
                data['ignore_switches'] = False
            print(data)
            game['rounds'].append(data)
        with open(annotation_path, 'w', encoding='utf8') as f:
            json.dump(game, f, indent=4)
    else:
        with open(annotation_path, 'r', encoding='utf8') as f:
            game = json.load(f)
    upload_game(game)

def vod_main():
    vods = get_annotate_vods()
    for v in vods:
        local_path = get_vod_path(v)
        if not os.path.exists(local_path):
            get_local_vod(v)
    for v in vods:
        get_rounds(v)


if __name__ == '__main__':
    import keras

    kf_cnn_model = load_kf_cnn_model()
    mid_model = load_mid_model()
    replay_model = load_replay_model()
    # kf_lstm_model = load_kf_lstm_model()
    player_cnn_model = load_player_cnn_model()
    ocr_model = load_player_ocr_model()
    # player_lstm_model = load_player_lstm_model()

    print('loaded model')
    #main()
    vod_main()