import h5py
import os
import random
import cv2
import numpy as np
import shutil

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, BOX_PARAMETERS
from annotator.api_requests import get_game_states
from annotator.utils import look_up_game_state, get_vod_path
from annotator.game_values import SPECTATOR_MODES, MAP_SET, COLOR_SET, FILM_FORMATS, SUBMAP_SET, GAME_SET


class GameGenerator(DataGenerator):
    identifier = 'game'
    resize_factor = 0.2
    broadcast_event_time_step = 0.1
    in_game_time_step = 2
    out_game_time_step = 4

    def __init__(self):
        super(GameGenerator, self).__init__()
        self.immutable_sets = {}
        #self.game_set = ['not_game', 'game', 'pause', 'replay', 'smaller_window']
        self.sets = {
            'game': GAME_SET,
            'map': ['n/a'] + MAP_SET,
            'submap': ['n/a'] + SUBMAP_SET,
            'attacking_side': ['n/a'] + ['neither', 'left', 'right'],
            'left_color': ['n/a'] + COLOR_SET,
            'right_color': ['n/a'] + COLOR_SET,
            'spectator_mode': ['n/a'] + SPECTATOR_MODES,
            'film_format': ['n/a'] + FILM_FORMATS,
            'left': ['n/a', 'not_zoom', 'zoom'],
            'right': ['n/a', 'not_zoom', 'zoom'],
        }

        self.image_width = 1280
        self.image_height = 720
        self.check_set_info()
        self.slots = [1]
        self.save_set_info()
        self.current_states = {}

    def save_set_info(self):
        super(GameGenerator, self).save_set_info()
        #path = os.path.join(self.training_directory, 'game_set.txt')
        #if not os.path.exists(path):
        #    with open(path, 'w', encoding='utf8') as f:
        #        for p in self.game_set:
        #            f.write('{}\n'.format(p))

    def figure_slot_params(self, r):
        self.slot_params = {}
        self.slot_params[1] = {'x': 0, 'y': 0}

    def lookup_data(self, time_point):
        d = {}
        beginning = False

        for k, current_state in self.current_states.items():
            try:
                while time_point >= self.states[k][self.current_states[k]]['end']:
                    self.current_states[k] += 1
                d[k] = self.states[k][self.current_states[k]]['status']
                if time_point == self.states[k][self.current_states[k]]['begin']:
                    beginning = True
            except IndexError:
                #if k == 'game':
                #    d[k] = self.game_set[0]
                #else:
                d[k] = self.sets[k][0]
        if d['game'] == 'not_game' or d['game'] == 'pause_player reaction':
            d['left'] = 'n/a'
            d['right'] = 'n/a'
            d['map'] = 'n/a'
            d['submap'] = 'n/a'
            d['left_color'] = 'n/a'
            d['right_color'] = 'n/a'
            d['attacking_side'] = 'n/a'
        elif d['game'] == 'replay':
            d['left'] = 'not_zoom'
            d['right'] = 'not_zoom'
        elif d['game'] == 'smaller_window':
            d['left'] = 'n/a'
            d['right'] = 'n/a'
        return d, beginning

    def check_status(self, d):
        for s in ['pause', 'replay', 'smaller_window']:
            if d['game'].startswith(s):
                return True
        if d['left'] == 'zoom' or d['right'] == 'zoom':
            return True
        return False

    def display_current_frame(self, frame, time_point, frame_ind):
        d, beginning = self.lookup_data(time_point)
        # print(frame.shape)
        print(d)
        print(beginning)
        box = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        # cv2.imshow('frame', box)
        # print(box.shape)
        # cv2.waitKey()
        cv2.imshow('game', box)
        #if d['game'] != 'game':
        #    self.ignored_indexes.append(index)

    def update_from_dict(self, return_dict):
        for k, v in return_dict.items():
            v_id, time_point = k
            d, ignore = v
            if self.process_index > len(self.indexes) - 1:
                continue
            index = self.indexes[self.process_index]
            if index < self.num_train:
                pre = 'train'
            else:
                pre = 'val'
                index -= self.num_train
            if ignore:
                self.ignored_indexes[pre].append(index)
                self.process_index += 1
                continue
            self.data["{}_img".format(pre)][index, ...] = d['img'][None]
            self.data["{}_vod".format(pre)][index] = v_id

            self.data["{}_time_point".format(pre)][index] = time_point

            #self.data["{}_game_label".format(pre)][index] = self.game_set.index(d['game'])
            for k, s in self.sets.items():
                self.data["{}_{}_label".format(pre, k)][index] = s.index(d[k])

            self.process_index += 1


    def process_frame(self, frame, time_point, frame_ind):
        if not self.generate_data:
            return
        d,beginning = self.lookup_data(time_point)
        if self.process_index > len(self.indexes) - 1:
            return
        index = self.indexes[self.process_index]
        if index < self.num_train:
            pre = 'train'
        else:
            pre = 'val'
            index -= self.num_train
        if beginning:
            self.ignored_indexes[pre].append(index)
            self.process_index += 1
            return
        # print(frame.shape)
        box = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        # cv2.imshow('frame', box)
        # print(box.shape)
        # cv2.waitKey()
        box = np.transpose(box, axes=(2, 0, 1))
        #if d['game'] != 'game':
        #    self.ignored_indexes.append(index)
        self.data["{}_img".format(pre)][index, ...] = box[None]
        self.data["{}_vod".format(pre)][index] = self.current_vod_id

        self.data["{}_time_point".format(pre)][index] = time_point

        #self.data["{}_game_label".format(pre)][index] = self.game_set.index(d['game'])
        for k, s in self.sets.items():
            self.data["{}_{}_label".format(pre, k)][index] = s.index(d[k])

        self.process_index += 1
        if self.debug:
            filename = '{}_{}.jpg'.format(' '.join(d.values()), index).replace(':', '')
            cv2.imwrite(os.path.join(self.training_directory, 'debug', pre,
                                     filename), np.transpose(box, axes=(1, 2, 0)))

    @property
    def minimum_time_step(self):
        if self.has_status:
            return self.broadcast_event_time_step
        else:
            return self.in_game_time_step

    def get_data(self, r):
        self.states = get_game_states(r['id'])
        self.has_status = False
        for label in [x for x in self.sets['game'] if x not in ['game', 'not_game']]:
            if label in [x['status'] for x in self.states['game']]:
                self.has_status = True
                break
        if not self.has_status:
            if 'zoom' in [x['status'] for x in self.states['left'] + self.states['right']]:
                self.has_status = True

        for k in self.sets.keys():
            self.current_states[k] = 0
        self.current_states['game'] = 0
        short_time_steps = [x for x in self.states['game'] if x['status'] not in ['not_game', 'game']]
        status_duration = round(
            sum([x['end'] - x['begin'] for x in self.states['game'] if x['status'] not in ['not_game', 'game']]), 1)
        already_done = set()
        for interval in self.states['left'] + self.states['right']:
            if interval['status'] == 'zoom' and interval['begin'] not in already_done:
                short_time_steps.append(interval)
                status_duration += round(interval['end'] - interval['begin'], 1)
                already_done.add(interval['begin'])
        game_boundaries = 0
        for i, interval in enumerate(self.states['game']):
            if interval['status'] == 'game':
                if self.states['game'][i-1]['status'] == 'not_game':
                    short_time_steps.append({'begin': interval['begin'] - 2, 'end': interval['begin'] + 2})
                    game_boundaries += 1
                elif i >= len(self.states['game']) - 1 or self.states['game'][i+1]['status'] == 'not_game':
                    short_time_steps.append({'begin': interval['end'] - 2, 'end': interval['end'] + 2})
                    game_boundaries += 1

        self.special_time_steps = {
            self.broadcast_event_time_step: sorted(short_time_steps, key=lambda x: x['begin']),
            self.in_game_time_step: sorted([x for x in self.states['game'] if x['status'] == 'game'],
                                           key=lambda x: x['begin']),
            self.out_game_time_step: sorted([x for x in self.states['game'] if x['status'] == 'not_game'],
                                            key=lambda x: x['begin']),
        }
        path = get_vod_path(r)
        stream = cv2.VideoCapture(path)
        fps = stream.get(cv2.CAP_PROP_FPS)
        self.expected_duration = round(stream.get(cv2.CAP_PROP_FRAME_COUNT) / fps, 1)
        stream.release()

        not_game_duration = sum(x['end'] - x['begin'] for x in self.states['game'] if x['status'] == 'not_game')
        game_duration = sum(x['end'] - x['begin'] for x in self.states['game'] if x['status'] == 'not_game')
        broadcast_event_duration = sum(x['end'] - x['begin'] for x in short_time_steps)
        game_duration -= broadcast_event_duration
        game_duration += game_boundaries * 2
        not_game_duration -= game_boundaries * 2
        extra = self.expected_duration - not_game_duration - game_duration - broadcast_event_duration
        not_game_duration += extra

        game_frame_count = int(game_duration / self.in_game_time_step)
        not_game_frame_count = int(not_game_duration / self.out_game_time_step)
        broadcast_event_frame_count = int(broadcast_event_duration / self.broadcast_event_time_step)
        num_frames = game_frame_count + not_game_frame_count + broadcast_event_frame_count
        print('NOT GAME', not_game_duration, not_game_frame_count)
        print('GAME', game_duration, game_frame_count)
        print('BROADCAST', broadcast_event_duration, broadcast_event_frame_count)
        print('DURATIONS', self.expected_duration, not_game_duration + game_duration + broadcast_event_duration)
        print('TOTAL FRAMES', num_frames)
        return num_frames

    def add_new_round_info(self, r):
        self.current_round_id = r['id']
        spec_dir = os.path.join(self.training_directory, r['event']['spectator_mode']['name'].lower())
        os.makedirs(spec_dir, exist_ok=True)
        old_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        self.hd5_path = os.path.join(spec_dir, '{}.hdf5'.format(r['id']))

        if os.path.exists(old_path):
            shutil.move(old_path, self.hd5_path)

        if os.path.exists(self.hd5_path):
            self.generate_data = False
            return

        num_frames = self.get_data(r)

        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train
        self.analyzed_rounds.append(r['id'])
        self.current_vod_id = r['id']
        self.generate_data = True
        self.figure_slot_params(r)

        self.indexes = random.sample(range(num_frames), num_frames)

        train_shape = (
            self.num_train, 3,
            int(self.image_height * self.resize_factor),
            int(self.image_width * self.resize_factor))
        val_shape = (
            self.num_val, 3,
            int(self.image_height * self.resize_factor),
            int(self.image_width * self.resize_factor))

        self.data = {}
        self.ignored_indexes = {}
        for pre in ['train', 'val']:
            self.ignored_indexes[pre] = []
            if pre == 'train':
                shape = train_shape
                count = self.num_train
            else:
                shape = val_shape
                count = self.num_val
            self.data["{}_img".format(pre)] = np.zeros(shape, dtype=np.uint8)
            self.data["{}_vod".format(pre)] = np.zeros((count,), dtype=np.int16)
            self.data["{}_time_point".format(pre)] = np.zeros((count,), dtype=np.float)
            #self.data["{}_game_label".format(pre)] = np.zeros((count,), dtype=np.uint8)
            for k, s in self.sets.items():
                self.data["{}_{}_label".format(pre, k)] = np.zeros((count,), dtype=np.uint8)

        self.process_index = 0

    def cleanup_round(self):
        if not self.generate_data:
            return
        for pre, ignored in self.ignored_indexes.items():
            if ignored:
                for k, v in self.data.items():
                    if pre not in k:
                        continue
                    self.data[k] = np.delete(v, ignored, axis=0)
        with h5py.File(self.hd5_path, mode='w') as hdf5_file:
            for k, v in self.data.items():
                #skip = False
                #for s in self.sets.keys():
                #    if s in k:
                #        skip = True
                #        break
                #if skip:
                #    continue
                hdf5_file.create_dataset(k, v.shape, v.dtype)
                hdf5_file[k][:] = v[:]
        #with h5py.File(self.detail_hd5_path, mode='w') as hdf5_file:
        #    for k, v in self.data.items():
        #        if 'game' in k:
        #            continue
        #        hdf5_file.create_dataset(k, v.shape, v.dtype)
        #        hdf5_file[k][:] = v[:]
        with open(self.rounds_analyzed_path, 'w') as f:
            for r in self.analyzed_rounds:
                f.write('{}\n'.format(r))
