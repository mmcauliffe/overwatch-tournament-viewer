import h5py
import os
import random
import cv2
import numpy as np
import pickle

from annotator.data_generation.classes.ctc import CTCDataGenerator, write_cache
from annotator.api_requests import get_player_states
from annotator.config import na_lab, sides, BOX_PARAMETERS
from annotator.game_values import COLOR_SET, PLAYER_CHARACTER_SET
from annotator.utils import look_up_player_state


class PlayerOCRGenerator(CTCDataGenerator):
    identifier = 'player_ocr'
    num_slots = 12
    time_step = 10
    num_variations = 1
    data_bytes = 6500

    def __init__(self, debug=False):
        super(PlayerOCRGenerator, self).__init__()
        self.label_set = PLAYER_CHARACTER_SET
        self.debug = debug
        if self.debug:
            os.makedirs(os.path.join(self.training_directory, 'debug', 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.training_directory, 'debug', 'val'), exist_ok=True)
        self.save_label_set()
        self.sets = {}
        self.save_set_info()

        params = BOX_PARAMETERS['O']['LEFT_NAME']
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        for side in sides:
            for i in range(6):
                self.slots.append((side, i))
        self.current_round_id = None
        self.check_set_info()

    def lookup_data(self, slot):
        return self.names[slot]

    def process_frame(self, frame, time_point):
        if not self.generate_data:
            return

        for s in self.slots:
            params = self.slot_params[s]
            sequence = self.lookup_data(s)

            variation_set = []
            while len(variation_set) < self.num_variations:
                x_offset = random.randint(-3, 3)
                y_offset = random.randint(-2, 2)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))

            x = params['x']
            y = params['y']

            for i, (x_offset, y_offset) in enumerate(variation_set):

                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                box = np.pad(box,((int(self.image_height/2), int(self.image_height/2)),(0,0),(0,0)), mode='constant', constant_values=0)
                box = np.transpose(box, axes=(2, 0, 1))
                train_check = random.random() <= 0.8
                if train_check:
                    pre = 'train'
                    index = self.previous_train_count
                    image_key = 'image-%09d' % index
                    label_key = 'label-%09d' % index
                    round_key = 'round-%09d' % index
                    time_point_key = 'time_point-%09d' % index
                    self.train_cache[image_key] = pickle.dumps(box)
                    self.train_cache[label_key] = sequence
                    self.train_cache[round_key] = str(self.current_round_id)
                    self.train_cache[time_point_key] = '{:.1f}'.format(time_point)
                    if len(self.train_cache) >= 4000:
                        write_cache(self.train_env, self.train_cache)
                        self.train_cache = {}
                    self.previous_train_count += 1
                else:
                    pre = 'val'
                    index = self.previous_val_count
                    image_key = 'image-%09d' % index
                    label_key = 'label-%09d' % index
                    round_key = 'round-%09d' % index
                    time_point_key = 'time_point-%09d' % index
                    self.val_cache[image_key] = pickle.dumps(box)
                    self.val_cache[label_key] = sequence
                    self.val_cache[round_key] = str(self.current_round_id)
                    self.val_cache[time_point_key] = '{:.1f}'.format(time_point)
                    if len(self.val_cache) >= 4000:
                        write_cache(self.val_env, self.val_cache)
                        self.val_cache = {}
                    self.previous_val_count += 1

                self.process_index += 1

    def add_new_round_info(self, r):
        super(PlayerOCRGenerator, self).add_new_round_info(r)
        if not self.generate_data:
            return
        self.states = get_player_states(r['id'])

        self.names = {}
        for slot in self.slots:
            self.names[slot] = self.states[slot[0]][str(slot[1])]['player'].lower()

    def figure_slot_params(self, r):
        left_params = BOX_PARAMETERS[r['stream_vod']['film_format']]['LEFT_NAME']
        right_params = BOX_PARAMETERS[r['stream_vod']['film_format']]['RIGHT_NAME']
        self.slot_params = {}
        for side in sides:
            if side == 'left':
                p = left_params
            else:
                p = right_params
            for i in range(6):
                self.slot_params[(side, i)] = {}
                self.slot_params[(side, i)]['x'] = p['X'] + (p['WIDTH'] + p['MARGIN']) * i
                self.slot_params[(side, i)]['y'] = p['Y']

    def calculate_map_size(self, rounds):
        num_frames = 0
        for r in rounds:
            for beg, end in r['sequences']:
                num_frames += int((end-beg) / self.time_step)
        overall = num_frames * self.data_bytes * 10 * self.num_variations * self.num_slots
        self.train_map_size = int(overall * 0.82)
        self.val_map_size = int(overall * 0.22)
        print(self.train_map_size)
