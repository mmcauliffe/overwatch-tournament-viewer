import h5py
import os
import random
import cv2
import numpy as np
import jamotools

from annotator.data_generation.classes.ctc import CTCDataGenerator
from annotator.api_requests import get_player_states
from annotator.config import na_lab, sides, BOX_PARAMETERS
from annotator.game_values import COLOR_SET, PLAYER_CHARACTER_SET, SPECTATOR_MODES
from annotator.utils import look_up_player_state


class PlayerOCRGenerator(CTCDataGenerator):
    identifier = 'player_ocr'
    num_slots = 12
    time_step = 0.3
    num_variations = 1
    resize_factor = 2
    usable_annotations = ['M', 'O']
    offset = 3
    duration = 5

    def __init__(self, debug=False):
        super(PlayerOCRGenerator, self).__init__()
        self.label_set = PLAYER_CHARACTER_SET
        self.debug=debug
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

    def lookup_data(self, slot, time_point):
        d = look_up_player_state(slot[0], slot[1], time_point, self.states)
        return [self.label_set.index(x) for x in jamotools.split_syllables(self.names[slot])], d['alive'], d['hero'], self.colors[slot]

    def display_current_frame(self, frame, time_point, frame_ind):
        for slot in self.slots:
            if isinstance(slot, (list, tuple)):
                slot_name = '_'.join(map(str, slot))
            else:
                slot_name = slot
            params = self.slot_params[slot]
            x = params['x']
            y = params['y']
            box = frame[y: y + self.image_height,
                  x: x + self.image_width]
            box = cv2.resize(box, (0, 0),fx=self.resize_factor, fy=self.resize_factor)
            cv2.imshow('{}_{}'.format(self.identifier, slot_name), box)

    def process_frame(self, frame, time_point, frame_ind):
        if not self.generate_data:
            return
        frame = frame['frame']
        #cv2.imshow('frame', frame)
        for s in self.slots:
            params = self.slot_params[s]
            ignore = False
            try:
                sequence, alive, hero, color = self.lookup_data(s, time_point)
                if hero == 'n/a':
                    ignore = True
            except IndexError:
                ignore = True

            variation_set = []
            while len(variation_set) < self.num_variations:
                x_offset = random.randint(-1, 1)
                y_offset = random.randint(-1, 1)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))

            x = params['x']
            y = params['y']

            for i, (x_offset, y_offset) in enumerate(variation_set):
                if self.process_index >= len(self.indexes) - 1:
                    continue
                index = self.indexes[self.process_index]
                if ignore:
                    self.ignored_indexes.append(index)
                if index < self.num_train:
                    pre = 'train'
                else:
                    pre = 'val'
                    index -= self.num_train
                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                box = cv2.resize(box, (0, 0),fx=self.resize_factor, fy=self.resize_factor)
                #box = np.pad(box,((int(self.image_height/2), int(self.image_height/2)),(0,0),(0,0)), mode='constant', constant_values=0)
                #if i == 0:
                    #cv2.imshow('gray_{}'.format(s), gray)
                    #cv2.imshow('bw_{}'.format(s), bw)
                box = np.transpose(box, axes=(2, 0, 1))
                self.data["{}_img".format(pre)][index, ...] = box[None]

                #self.train_mean += box / self.hdf5_file['train_img'].shape[0]
                sequence_length = len(sequence)
                self.data['{}_spectator_mode_label'.format(pre)][index] = SPECTATOR_MODES.index(self.spec_mode)
                if sequence:
                    self.data["{}_label_sequence_length".format(pre)][index] = sequence_length
                    self.data["{}_label_sequence".format(pre)][index, 0:len(sequence)] = sequence
                    self.data["{}_round".format(pre)][index] = self.current_round_id
                    self.data["{}_time_point".format(pre)][index] = time_point
                self.process_index += 1
        #cv2.waitKey(0)

    def add_new_round_info(self, r):
        self.current_round_id = r['id']
        import shutil
        spec_dir = os.path.join(self.training_directory, r['spectator_mode'].lower())
        os.makedirs(spec_dir, exist_ok=True)
        old_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        self.hd5_path = os.path.join(spec_dir, '{}.hdf5'.format(r['id']))
        if os.path.exists(old_path):
            shutil.move(old_path, self.hd5_path)
            self.generate_data = False
            return
        if os.path.exists(self.hd5_path) or r['annotation_status'] not in self.usable_annotations:
            self.generate_data = False
            return
        num_frames = 0
        self.spec_mode = r['spectator_mode'].lower()
        for beg, end in r['sequences']:
            if end - beg < self.offset + self.duration:
                continue
            if end > beg + self.offset + self.duration:
                end = beg + self.offset + self.duration
            beg += self.offset
            print(beg, end)
            expected_frame_count = int((end - beg) / self.time_step)
            num_frames += (int(expected_frame_count) + 1) * self.num_slots
        print(num_frames)
        num_frames *= self.num_variations
        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train
        self.current_round_id = r['id']
        self.generate_data = True
        self.figure_slot_params(r)
        self.analyzed_rounds.append(r['id'])
        self.spec_mode = r['spectator_mode'].lower()
        self.indexes = random.sample(range(num_frames), num_frames)
        self.ignored_indexes = []

        self.left_color = r['game']['left_team']['color'].lower()
        self.right_color = r['game']['right_team']['color'].lower()
        train_shape = (
            self.num_train,3, int(self.image_height * self.resize_factor),
            int(self.image_width * self.resize_factor))
        val_shape = (
            self.num_val, 3,  int(self.image_height * self.resize_factor),
            int(self.image_width * self.resize_factor))

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
            self.data["{}_spectator_mode_label".format(pre)] = np.zeros( (count,), dtype=np.uint8)
            self.data["{}_label_sequence".format(pre)] = np.zeros((count, self.max_sequence_length),
                                          dtype=np.int16)
            self.data["{}_label_sequence".format(pre)][:, :] = len(self.label_set)
            self.data["{}_label_sequence_length".format(pre)] = np.ones((count,), dtype=np.uint8)

        self.process_index = 0
        self.states = get_player_states(r['id'])

        self.names = {}
        self.colors = {}
        for slot in self.slots:
            self.names[slot] = self.states[slot[0]][str(slot[1])]['player'].lower()
            if slot[0] == 'left':
                self.colors[slot] = self.left_color
            else:
                self.colors[slot] = self.right_color

    def figure_slot_params(self, r):
        film_format = r['stream_vod']['film_format']['code']
        left_params = BOX_PARAMETERS[film_format]['LEFT_NAME']
        right_params = BOX_PARAMETERS[film_format]['RIGHT_NAME']
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
        print(self.slot_params)

    def cleanup_round(self):
        if not self.generate_data:
            return
        if self.ignored_indexes:
            for k, v in self.data.items():
                self.data[k] = np.delete(v, self.ignored_indexes, axis=0)
        super(PlayerOCRGenerator, self).cleanup_round()