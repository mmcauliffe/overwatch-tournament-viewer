import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, BOX_PARAMETERS
from annotator.api_requests import get_game_states
from annotator.utils import look_up_game_state, get_vod_path
from annotator.game_values import SPECTATOR_MODES, MAP_SET, COLOR_SET, FILM_FORMATS


class GameGenerator(DataGenerator):
    identifier = 'game'
    resize_factor = 0.2
    time_step = 0.1
    secondary_time_step = 1

    def __init__(self):
        super(GameGenerator, self).__init__()
        self.immutable_sets = {}
        self.sets = {
            'game': ['not_game', 'game', 'pause', 'replay','smaller_window'],
            'map': ['n/a'] + MAP_SET,
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

    def figure_slot_params(self, r):
        self.slot_params = {}
        self.slot_params[1] = {'x': 0, 'y': 0}

    def lookup_data(self, slot, time_point):
        d = {}
        for k, current_state in self.current_states.items():
            try:
                if time_point >= self.states[k][current_state]['end'] and current_state < len(self.states[k]):
                    self.current_states[k] += 1
                d[k] = self.states[k][self.current_states[k]]['status']
            except IndexError:
                d[k] = self.sets[k][0]
        return d

    def check_status(self, d):
        if d['game'] in ['pause', 'replay', 'smaller_window']:
            return True
        if d['left'] == 'zoom' or d['right'] == 'zoom':
            return True
        return False

    def process_frame(self, frame, time_point, frame_ind):
        if not self.generate_data:
            return
        d = self.lookup_data(1, time_point)
        params = self.slot_params[1]
        x = params['x']
        y = params['y']
        if self.process_index > len(self.indexes) - 1:
            return
        index = self.indexes[self.process_index]
        if index < self.num_train:
            pre = 'train'
        else:
            pre = 'val'
            index -= self.num_train
        #print(frame.shape)
        box = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        #cv2.imshow('frame', box)
        #print(box.shape)
        #cv2.waitKey()
        box = np.transpose(box, axes=(2, 0, 1))
        self.data["{}_img".format(pre)][index, ...] = box[None]
        self.data["{}_round".format(pre)][index] = self.current_round_id

        self.data["{}_time_point".format(pre)][index] = time_point

        for k, s in self.sets.items():
            self.data["{}_{}_label".format(pre, k)][index] = s.index(d[k])

        self.process_index += 1
        if self.debug:

            filename = '{}_{}.jpg'.format(' '.join(d.values()), index).replace(':', '')
            cv2.imwrite(os.path.join(self.training_directory, 'debug', pre,
                                 filename), np.transpose(box, axes=(1,2,0)))

    @property
    def minimum_time_step(self):
        if self.has_status:
            return self.time_step
        else:
            return self.secondary_time_step

    def add_new_round_info(self, r):
        self.current_round_id = r['id']
        self.hd5_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        self.states = get_game_states(r['id'])
        self.has_status = False
        for label in ['pause', 'replay', 'smaller_window']:
            if label in [x['status'] for x in self.states['game']]:
                self.has_status = True
                break
        if not self.has_status:
            if 'zoom' in [x['status'] for x in self.states['left'] + self.states['right']]:
                self.has_status = True

        for k in self.sets.keys():
            self.current_states[k] = 0
        if os.path.exists(self.hd5_path):
            self.generate_data = False
            return

        num_frames = 0
        short_time_steps = [x for x in self.states['game'] if x['status'] not in ['not_game', 'game']]
        status_duration = round(sum([x['end'] - x['begin'] for x in self.states['game'] if x['status'] not in ['not_game', 'game']]), 1)
        already_done = set()
        for interval in self.states['left'] + self.states['right']:
            if interval['status'] == 'zoom' and interval['begin'] not in already_done:
                short_time_steps.append(interval)
                status_duration += round(interval['end'] - interval['begin'], 1)
                already_done.add(interval['begin'])
        self.short_time_steps = sorted(short_time_steps, key= lambda x:x['begin'])
        path = get_vod_path(r)
        stream = cv2.VideoCapture(path)
        fps = stream.get(cv2.CAP_PROP_FPS)
        expected_duration = round(stream.get(cv2.CAP_PROP_FRAME_COUNT) / fps, 1)
        stream.release()

        normal_duration = expected_duration - status_duration
        normal_frame_count = int(normal_duration / self.secondary_time_step)
        status_frame_count = int(status_duration / self.time_step)
        num_frames += normal_frame_count + status_frame_count
        print('NORMAL', normal_duration, normal_frame_count)
        print('STATUS', status_duration, status_frame_count)
        print('TOTAL FRAMES', num_frames)

        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train
        self.analyzed_rounds.append(r['id'])
        self.current_round_id = r['id']
        self.generate_data = True
        self.figure_slot_params(r)

        self.indexes = random.sample(range(num_frames), num_frames)

        train_shape = (
        self.num_train, 3, int(self.image_height * self.resize_factor), int(self.image_width * self.resize_factor))
        val_shape = (
        self.num_val, 3, int(self.image_height * self.resize_factor), int(self.image_width * self.resize_factor))

        self.data = {}
        for pre in ['train', 'val']:
            if pre == 'train':
                shape = train_shape
                count = self.num_train
            else:
                shape = val_shape
                count = self.num_val
            self.data["{}_img".format(pre)] = np.zeros(shape, dtype=np.uint8)
            self.data["{}_round".format(pre)] = np.zeros((count,), dtype=np.int16)
            self.data["{}_time_point".format(pre)] = np.zeros((count,), dtype=np.float)
            for k, s in self.sets.items():
                self.data["{}_{}_label".format(pre, k)] =  np.zeros((count,), dtype=np.uint8)

        self.process_index = 0

