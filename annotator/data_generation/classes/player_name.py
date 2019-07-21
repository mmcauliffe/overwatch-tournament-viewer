import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.ctc import CTCDataGenerator
from annotator.api_requests import get_player_states
from annotator.config import na_lab, sides, BOX_PARAMETERS
from annotator.game_values import COLOR_SET, PLAYER_CHARACTER_SET
from annotator.utils import look_up_player_state


class PlayerOCRGenerator(CTCDataGenerator):
    identifier = 'player_ocr'
    num_slots = 12
    time_step = 10
    num_variations = 1

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
        return [self.label_set.index(x) for x in self.names[slot]], d['alive'], self.colors[slot]

    def process_frame(self, frame, time_point):
        if not self.generate_data:
            return
        #cv2.imshow('frame', frame)
        for s in self.slots:
            params = self.slot_params[s]
            sequence, alive, color = self.lookup_data(s, time_point)

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
                index = self.indexes[self.process_index]
                if index < self.num_train:
                    pre = 'train'
                else:
                    pre = 'val'
                    index -= self.num_train
                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                #if i == 0:
                    #cv2.imshow('gray_{}'.format(s), gray)
                    #cv2.imshow('bw_{}'.format(s), bw)
                if self.debug:
                    name = self.names[s]
                    if name.startswith('blas'):
                        name = 'blase'
                    cv2.imwrite(os.path.join(self.training_directory, 'debug', pre, '{}_{}.jpg'.format(name, self.process_index)), box)
                box = np.swapaxes(box, 1, 0)
                self.hdf5_file["{}_img".format(pre)][index, ...] = box[None]

                #self.train_mean += box / self.hdf5_file['train_img'].shape[0]
                sequence_length = len(sequence)
                if sequence:
                    self.hdf5_file["{}_label_sequence_length".format(pre)][index] = sequence_length
                    self.hdf5_file["{}_label_sequence".format(pre)][index, 0:len(sequence)] = sequence
                    self.hdf5_file["{}_round".format(pre)][index] = self.current_round_id
                self.process_index += 1
        #cv2.waitKey(0)

    def add_new_round_info(self, r):
        self.current_round_id = r['id']
        self.hd5_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        if os.path.exists(self.hd5_path):
            self.generate_data = False
            return
        num_frames = 0
        for beg, end in r['sequences']:
            print(beg, end)
            expected_frame_count = int((end - beg) / self.time_step)
            num_frames += (int(expected_frame_count) + 1) * self.num_slots

        num_frames *= self.num_variations
        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train
        self.current_round_id = r['id']
        self.generate_data = True
        self.figure_slot_params(r)
        self.analyzed_rounds.append(r['id'])
        self.spec_mode = r['spectator_mode'].lower()
        self.indexes = random.sample(range(num_frames), num_frames)

        self.left_color = r['game']['left_team']['color'].lower()
        self.right_color = r['game']['right_team']['color'].lower()
        train_shape = (
            self.num_train, int(self.image_width * self.resize_factor), int(self.image_height * self.resize_factor), 3)
        val_shape = (
            self.num_val, int(self.image_width * self.resize_factor), int(self.image_height * self.resize_factor), 3)
        self.hdf5_file = h5py.File(self.hd5_path, mode='w')
        #self.hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
        #self.train_mean = np.zeros(train_shape[1:], np.float32)
        for pre in ['train', 'val']:
            if pre == 'train':
                shape = train_shape
                count = self.num_train
            else:
                shape = val_shape
                count = self.num_val
            self.hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8,
                                          maxshape=(None, shape[1], shape[2], shape[3]))
            self.hdf5_file.create_dataset("{}_label_sequence".format(pre), (count, self.max_sequence_length),
                                          np.uint32, maxshape=(None, self.max_sequence_length),
                                          fillvalue=len(self.label_set))
            self.hdf5_file.create_dataset("{}_label_sequence_length".format(pre), (count,), np.uint8,
                                          maxshape=(None,), fillvalue=1)
            self.hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32, maxshape=(None,))

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
        print(self.slot_params)