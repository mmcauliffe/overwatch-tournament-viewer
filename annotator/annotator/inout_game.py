import os
import json
import cv2
import numpy as np
import re
import keras
import itertools
import time
from collections import defaultdict, Counter

from annotator.config import na_lab, sides, BOX_PARAMETERS
from annotator.utils import get_local_vod, \
    get_local_path,  FileVideoStream, Empty, get_vod_path
from annotator.api_requests import get_annotate_vods_in_out_game, upload_annotated_in_out_game

frames_per_seq = 100

working_dir = r'E:\Data\Overwatch\models'
player_model_dir = os.path.join(working_dir, 'player_status')
player_ocr_model_dir = os.path.join(working_dir, 'player_ocr')
kf_ctc_model_dir = os.path.join(working_dir, 'kill_feed_ctc')
game_model_dir = os.path.join(working_dir, 'game_cnn')
mid_model_dir = os.path.join(working_dir, 'mid')

annotation_dir = r'E:\Data\Overwatch\annotations'
oi_annotation_dir = r'E:\Data\Overwatch\oi_annotations'


def extract_info(v):
    owl_mapping = {'DAL': 'Dallas Fuel',
                   'PHI': 'Philadelphia Fusion', 'SEO': 'Seoul Dynasty',
                   'LDN': 'London Spitfire', 'SFS': 'San Francisco Shock', 'HOU': 'Houston Outlaws',
                   'BOS': 'Boston Uprising', 'VAL': 'Los Angeles Valiant', 'GLA': 'Los Angeles Gladiators',
                   'FLA': 'Florida Mayhem', 'SHD': 'Shanghai Dragons', 'NYE': 'New York Excelsior'}
    channel = v['channel']['name']
    info = {}
    if channel.lower() in ['overwatchcontenders', 'overwatchcontendersbr']:
        pattern = r'''(?P<team_one>[-\w ']+) (vs|V) (?P<team_two>[-\w ']+) \| (?P<desc>[\w ]+) Game (?P<game_num>\d) \| ((?P<sub>[\w :]+) \| )?(?P<main>[\w ]+)'''
        m = re.match(pattern, v['title'])
        if m is not None:
            print(m.groups())
            info['team_one'] = m.group('team_one')
            info['team_two'] = m.group('team_two')
            info['description'] = m.group('desc')
            info['game_number'] = m.group('game_num')
            sub = m.group('sub')
            main = m.group('main')
            info['event'] = main
            if sub is not None:
                info['event'] += ' - ' + sub
    elif channel.lower() == 'overwatchleague':
        pattern = r'Game (\d+) (\w+) @ (\w+) \| ([\w ]+)'
        m = re.match(pattern, v['title'])
        if m is not None:

            game_number, team_one, team_two, desc = m.groups()
            info['team_one'] = owl_mapping[team_one]
            info['team_two'] = owl_mapping[team_two]
            info['game_number'] = game_number
    elif channel.lower() =='owlettournament':
        pattern = r'''.* - (?P<team_one>[-\w ']+) (vs[.]?|V) (?P<team_two>[-\w ']+)'''

        m = re.match(pattern, v['title'])
        if m is not None:
            info['team_one'] = m.group('team_one')
            info['team_two'] = m.group('team_two')
    elif channel.lower() =='owlet tournament':
        pattern = r'''.*: (?P<team_one>[-\w ']+) (vs[.]?|V) (?P<team_two>[-\w ']+)'''

        m = re.match(pattern, v['title'])
        if m is not None:
            info['team_one'] = m.group('team_one')
            info['team_two'] = m.group('team_two')

    return info


class InGameAnnotator(object):
    time_step = 1
    resize_factor = 0.5

    def __init__(self, model_directory, film_format):
        self.model_directory = model_directory
        final_output_weights = os.path.join(self.model_directory, 'game_weights.h5')
        final_output_json = os.path.join(self.model_directory, 'game_model.json')
        with open(final_output_json, 'r') as f:
            loaded_model_json = f.read()
        self.model = keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(final_output_weights)

        self.params = BOX_PARAMETERS[film_format]['GAME']
        self.shape = (frames_per_seq, int(self.params['HEIGHT'] * self.resize_factor), int(self.params['WIDTH'] * self.resize_factor), 3)
        self.to_predict = np.zeros(self.shape, dtype=np.uint8)
        self.process_index = 0
        self.begin_time = 0

        self.status = []

    def process_frame(self, frame):
        box = frame[self.params['Y']: self.params['Y'] + self.params['HEIGHT'],
              self.params['X']: self.params['X'] + self.params['WIDTH']]
        box = cv2.resize(box, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        self.to_predict[self.process_index, ...] = box[None]
        self.process_index += 1

        if self.process_index == frames_per_seq:
            self.annotate()
            self.to_predict = np.zeros(self.shape, dtype=np.uint8)
            self.process_index = 0
            self.begin_time += frames_per_seq * self.time_step

    def annotate(self):
        if self.process_index == 0:
            return
        labels = ['not_in_game', 'in_game']
        lstm_output = self.model.predict([self.to_predict[None]])
        label_inds = lstm_output.argmax(axis=2)
        for t_ind in range(frames_per_seq):
            current_time = self.begin_time + (t_ind * self.time_step)
            lstm_label = labels[label_inds[0, t_ind]]
            if len(self.status) == 0:
                self.status.append({'begin': 0, 'end': 0, 'status': lstm_label})
            else:
                if lstm_label == self.status[-1]['status']:
                    self.status[-1]['end'] = current_time
                else:
                    self.status.append(
                        {'begin': current_time, 'end': current_time, 'status': lstm_label})


def annotate_game_or_not(v):
    game_annotator = InGameAnnotator(game_model_dir, v['film_format'])
    fvs = FileVideoStream(get_vod_path(v), 0, 0, game_annotator.time_step).start()
    time.sleep(5)
    print('begin ingame/outofgame processing')
    while True:
        try:
            frame, time_point = fvs.read()
        except Empty:
            break
        print(time_point)
        game_annotator.process_frame(frame)
    game_annotator.annotate()
    return game_annotator.status

def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


class MidAnnotator(object):
    resize_factor = 0.5
    time_step = 0.1

    def __init__(self, model_directory, film_format):
        self.film_format = film_format
        self.model_directory = model_directory
        final_output_weights = os.path.join(self.model_directory, 'mid_weights.h5')
        final_output_json = os.path.join(self.model_directory, 'mid_model.json')
        with open(final_output_json, 'r') as f:
            loaded_model_json = f.read()
        self.model = keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(final_output_weights)
        self.params = BOX_PARAMETERS[self.film_format]['MID']
        self.max_num_sequences = 10
        self.shape = (self.max_num_sequences, frames_per_seq, int(self.params['HEIGHT'] * self.resize_factor), int(self.params['WIDTH'] * self.resize_factor), 3)
        self.to_predict = np.zeros(self.shape, dtype=np.uint8)
        self.process_index = 0
        self.num_sequences = 0
        self.begin_time = 0
        mid_set_files = {
            'overtime': os.path.join(self.model_directory, 'overtime_set.txt'),
            'point_status': os.path.join(self.model_directory, 'point_status_set.txt'),
        }

        mid_end_set_files = {
            'attacking_color': os.path.join(self.model_directory, 'attacking_color_set.txt'),
            'map': os.path.join(self.model_directory, 'map_set.txt'),
            'map_mode': os.path.join(self.model_directory, 'map_mode_set.txt'),
            'round_number': os.path.join(self.model_directory, 'round_number_set.txt'),
            #'spectator_mode': os.path.join(self.model_directory, 'spectator_mode_set.txt'),
        }

        self.sets = {}

        for k, v in mid_set_files.items():
            self.sets[k] = load_set(v)

        self.end_sets = {}
        for k, v in mid_end_set_files.items():
            self.end_sets[k] = load_set(v)

        self.statuses = {k: [] for k in list(self.sets.keys()) + list(self.end_sets.keys())}

    def process_frame(self, frame):
        box = frame[self.params['Y']: self.params['Y'] + self.params['HEIGHT'],
              self.params['X']: self.params['X'] + self.params['WIDTH']]
        box = cv2.resize(box, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        cv2.imshow('frame', box)
        cv2.waitKey()
        self.to_predict[self.num_sequences, self.process_index, ...] = box[None]
        self.process_index += 1
        if self.process_index == frames_per_seq:
            self.num_sequences += 1
            self.process_index = 0
            if self.num_sequences == self.max_num_sequences:
                self.annotate()
                self.to_predict = np.zeros(self.shape, dtype=np.uint8)
                self.num_sequences = 0
                self.begin_time += self.max_num_sequences * frames_per_seq * self.time_step
                print(self.begin_time)
                for k,v in self.generate_round_properties().items():
                    print(k, v)

    def annotate(self):
        if self.num_sequences == 0:
            return
        lstm_output = self.model.predict([self.to_predict])
        for i in range(self.num_sequences):
            for output_ind, (output_key, s) in enumerate(list(self.sets.items()) + list(self.end_sets.items())):
                if output_key in self.end_sets:
                    label_inds = lstm_output[output_ind].argmax(axis=1)
                    lstm_label = s[label_inds[i]]
                    #print(output_key, lstm_label)
                    if lstm_label == 'n/a' and output_key != 'attacking_color':
                        continue
                    if len(self.statuses[output_key]) == 0:
                        self.statuses[output_key].append(
                            {'begin': self.begin_time + self.time_step * frames_per_seq * (i), 'end': self.begin_time + self.time_step * frames_per_seq * (i+1), 'status': lstm_label})
                    else:
                        if lstm_label == self.statuses[output_key][-1]['status']:
                            self.statuses[output_key][-1]['end'] = self.begin_time + self.time_step * frames_per_seq * (i+1)
                        else:
                            self.statuses[output_key].append(
                                {'begin': self.begin_time + self.time_step * frames_per_seq * (i), 'end': self.begin_time + self.time_step * frames_per_seq * (i+1),
                                 'status': lstm_label})
                else:
                    label_inds = lstm_output[output_ind].argmax(axis=2)
                    for t_ind in range(frames_per_seq):
                        current_time = self.begin_time + (t_ind * self.time_step) +self.time_step * frames_per_seq *i
                        lstm_label = s[label_inds[i, t_ind]]
                        #print(output_key, lstm_label)
                        if lstm_label != 'n/a':
                            if len(self.statuses[output_key]) == 0:
                                self.statuses[output_key].append({'begin': 0, 'end': 0, 'status': lstm_label})
                            else:
                                if lstm_label == self.statuses[output_key][-1]['status']:
                                    self.statuses[output_key][-1]['end'] = current_time
                                else:
                                    self.statuses[output_key].append(
                                        {'begin': current_time, 'end': current_time, 'status': lstm_label})
            #cv2.imshow('frame', self.to_predict[i][0])
            #cv2.waitKey(0)

    def generate_round_properties(self):
        actual_overtime = []
        for i, r in enumerate(self.statuses['overtime']):
            if r['status'] == 'not_overtime' and r['end'] - r['begin'] < 2:
                continue
            if len(actual_overtime) and r['status'] == actual_overtime[-1]['status']:
                actual_overtime[-1]['end'] = r['end']
            else:
                if len(actual_overtime) and actual_overtime[-1]['end'] != r['begin']:
                    actual_overtime[-1]['end'] = r['begin']
                actual_overtime.append(r)
        out_props = {}
        actual_points = []
        for i, r in enumerate(self.statuses['point_status']):
            if r['end'] - r['begin'] < 2:
                continue
            if len(actual_points) and r['status'] == actual_points[-1]['status']:
                actual_points[-1]['end'] = r['end']
            else:
                if len(actual_points) and actual_points[-1]['end'] != r['begin']:
                    actual_points[-1]['end'] = r['begin']
                actual_points.append(r)

        for k in ['attacking_color', 'map', 'map_mode', 'round_number']:
            counts = defaultdict(float)
            for r in self.statuses[k]:
                counts[r['status']] += r['end'] - r['begin']
            print(k, counts)
            out_props[k] = max(counts, key=lambda x: counts[x])
        out_props['overtime'] = actual_overtime
        out_props['points'] = actual_points
        return out_props


def analyze_ingames(vods):
    game_dir = os.path.join(oi_annotation_dir, 'to_check')
    os.makedirs(game_dir, exist_ok=True)
    for v in vods:
        print(v)
        info= extract_info(v)
        if not info:
            continue
        data = {'vod_id': v['id'], 'rounds': [], 'team_one': info['team_one'], 'team_two': info['team_two']}
        game_status = annotate_game_or_not(v)
        for g in game_status:
            if g['status'] == 'in_game':
                dur = g['end'] - g['begin']
                if dur < 20:
                    continue
                data['rounds'].append({'begin': g['begin'], 'end': g['end']})
        print(data)
        upload_annotated_in_out_game(data)
        error


def vod_main():
    #test()
    #error
    vods = get_annotate_vods_in_out_game()
    for v in vods:
        local_path = get_vod_path(v)
        if not os.path.exists(local_path):
            get_local_vod(v)

    analyze_ingames(vods)


if __name__ == '__main__':


    print('loaded model')
    # main()
    vod_main()
