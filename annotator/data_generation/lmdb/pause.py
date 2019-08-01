import h5py
import os
import random
import cv2
import numpy as np
import sys

from annotator.data_generation.lmdb.base import DataGenerator, write_cache
from annotator.config import na_lab, BOX_PARAMETERS
from annotator.utils import look_up_round_state
from annotator.api_requests import get_round_states
from annotator.game_values import COLOR_SET, MAP_SET, MAP_MODE_SET, SPECTATOR_MODES
import pickle


class PauseStatusGenerator(DataGenerator):
    identifier = 'pause'
    resize_factor = 0.5
    num_variations = 3
    time_step = 1
    data_bytes = 5800

    def __init__(self, debug=False):
        super(PauseStatusGenerator, self).__init__(debug=debug)
        self.immutable_sets = {}
        self.sets = {
            'pause': ['not_pause', 'pause'],
        }
        params = BOX_PARAMETERS['O']['PAUSE']
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        self.check_set_info()
        self.slots = [1]
        self.save_set_info()

    def figure_slot_params(self, r):
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['PAUSE']
        self.slot_params = {}
        self.slot_params[1] = {'x': params['X'], 'y': params['Y']}

    def lookup_data(self, slot, time_point):
        d = look_up_round_state(time_point, self.states)
        return d

    def process_frame(self, frame, time_point):
        if not self.generate_data:
            return
        for slot in self.slots:
            d = self.lookup_data(slot, time_point)
            params = self.slot_params[slot]
            x = params['x']
            y = params['y']
            variation_set = []
            while len(variation_set) < self.num_variations:
                x_offset = random.randint(-4, 4)
                y_offset = random.randint(-4, 4)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))
            for i, (x_offset, y_offset) in enumerate(variation_set):
                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                if self.resize_factor != 1:
                    box = cv2.resize(box, (0, 0), fx=self.resize_factor, fy=self.resize_factor)

                box = np.transpose(box, axes=(2, 0, 1))
                train_check = random.random() <= 0.8
                if train_check:
                    pre = 'train'
                    index = self.previous_train_count
                    image_key = 'image-%09d' % index
                    round_key = 'round-%09d' % index
                    time_point_key = 'time_point-%09d' % index
                    self.train_cache[image_key] = pickle.dumps(box)
                    self.train_cache[round_key] = str(self.current_round_id)
                    self.train_cache[time_point_key] = '{:.1f}'.format(time_point)
                    for k in self.sets.keys():
                        key = '%s-%09d'% (k, index)
                        self.train_cache[key] = d[k]
                    self.previous_train_count += 1
                else:
                    pre = 'val'
                    index = self.previous_val_count
                    image_key = 'image-%09d' % index
                    round_key = 'round-%09d' % index
                    time_point_key = 'time_point-%09d' % index
                    self.val_cache[image_key] = pickle.dumps(box)
                    self.val_cache[round_key] = str(self.current_round_id)
                    self.val_cache[time_point_key] = '{:.1f}'.format(time_point)
                    for k in self.sets.keys():
                        key = '%s-%09d'% (k, index)
                        self.val_cache[key] = d[k]
                    self.previous_val_count += 1

                self.process_index += 1
                if self.process_index % 1000 == 0:
                    write_cache(self.train_env, self.train_cache)
                    self.train_cache = {}
                    write_cache(self.val_env, self.val_cache)
                    self.val_cache = {}
                if self.debug:
                    #if d['replay'] == 'replay':
                    #    self.display_current_frame(frame, time_point)
                    #    cv2.waitKey()
                    if self.identifier in self.sets:
                        if d[self.identifier].startswith(self.identifier):
                            filename = '{}_{}.jpg'.format(' '.join([d[x] for x in self.sets.keys()]), index).replace(':', '')
                            cv2.imwrite(os.path.join(self.training_directory, 'debug', pre,
                                                 filename), np.transpose(box, axes=(1,2,0)))
                    else:
                        filename = '{}_{}.jpg'.format(' '.join([d[x] for x in self.sets.keys()]), index).replace(':', '')
                        cv2.imwrite(os.path.join(self.training_directory, 'debug', pre,
                                             filename), np.transpose(box, axes=(1,2,0)))

    def add_new_round_info(self, r):
        super(PauseStatusGenerator, self).add_new_round_info(r)
        if not self.generate_data:
            return
        self.states = get_round_states(r['id'])
        for s in self.states['pauses']:
            if s['status'] == 'paused':
                break
        else:
            self.generate_data = False

    def calculate_map_size(self, rounds):
        num_frames = 0
        for r in rounds:
            states = get_round_states(r['id'])
            for s in states['pauses']:
                if s['status'] == 'paused':
                    break
            else:
                continue
            num_frames +=int((r['end'] - r['begin']) / self.time_step)
        overall = num_frames * self.data_bytes * 10 * self.num_variations
        self.train_map_size = int(overall * 0.82)
        self.val_map_size = int(overall * 0.22)
        print(self.train_map_size)
