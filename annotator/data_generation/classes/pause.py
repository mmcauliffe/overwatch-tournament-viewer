import h5py
import os
import random
import cv2
import numpy as np
import sys

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, BOX_PARAMETERS
from annotator.utils import look_up_single_round_state
from annotator.api_requests import get_round_states
import pickle


class PauseStatusGenerator(DataGenerator):
    identifier = 'pause'
    resize_factor = 0.5
    num_variations = 3
    time_step = 0.1

    def __init__(self, debug=False):
        super(PauseStatusGenerator, self).__init__(debug=debug)
        self.immutable_sets = {}
        self.sets = {
            self.identifier: ['not_'+ self.identifier, self.identifier],
        }
        params = BOX_PARAMETERS['2'][self.identifier.upper()]
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        self.check_set_info()
        self.slots = [1]
        self.save_set_info()

    def figure_slot_params(self, r):
        print(r)
        print(BOX_PARAMETERS[r['game']['match']['event']['film_format']])
        params = BOX_PARAMETERS[r['game']['match']['event']['film_format']][self.identifier.upper()]
        self.slot_params = {}
        self.slot_params[1] = {'x': params['X'], 'y': params['Y']}

    def lookup_data(self, slot, time_point):
        data = look_up_single_round_state(time_point, self.states, self.identifier)
        return data

    def process_frame(self, frame, time_point, frame_ind):
        if not self.generate_data:
            return
        for slot in self.slots:
            d = self.lookup_data(slot, time_point)
            if d[self.identifier] != self.identifier and frame_ind % 10 != 0:
                return
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
                if self.process_index > len(self.indexes) - 1:
                    continue
                index = self.indexes[self.process_index]
                if index < self.num_train:
                    pre = 'train'
                else:
                    pre = 'val'
                    index -= self.num_train
                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                if self.resize_factor != 1:
                    box = cv2.resize(box, (0, 0), fx=self.resize_factor, fy=self.resize_factor)

                box = np.transpose(box, axes=(2, 0, 1))
                self.hdf5_file["{}_img".format(pre)][index, ...] = box[None]
                self.hdf5_file["{}_round".format(pre)][index] = self.current_round_id

                self.hdf5_file["{}_time_point".format(pre)][index] = time_point

                for k, s in self.sets.items():
                    self.hdf5_file["{}_{}_label".format(pre, k)][index] = s.index(d[k])



                self.process_index += 1
                if self.debug:

                    filename = '{}_{}.jpg'.format(' '.join(d.values()), index).replace(':', '')
                    cv2.imwrite(os.path.join(self.training_directory, 'debug', pre,
                                         filename), np.transpose(box, axes=(1,2,0)))

    def add_new_round_info(self, r):
        self.current_round_id = r['id']
        self.hd5_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        if os.path.exists(self.hd5_path):
            self.generate_data = False
            return
        self.states = get_round_states(r['id'])
        not_duration = 0
        expected_duration = 0
        for s in self.states[self.identifier]:
            if s['status'] == self.identifier:
                expected_duration += s['end'] - s['begin']
            else:
                not_duration += s['end'] - s['begin']
        self.generate_data = expected_duration > 0

        if not self.generate_data:
            return
        self.get_data(r)
        num_frames = int(expected_duration / self.time_step) * self.num_variations
        num_frames += int(not_duration / (self.time_step * 2)) * self.num_variations

        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train
        self.analyzed_rounds.append(r['id'])
        self.current_round_id = r['id']
        self.generate_data = True
        self.figure_slot_params(r)

        self.indexes = random.sample(range(num_frames), num_frames)

        train_shape = (self.num_train, 3, int(self.image_height *self.resize_factor), int(self.image_width*self.resize_factor))
        val_shape = (self.num_val, 3, int(self.image_height *self.resize_factor), int(self.image_width*self.resize_factor))
        self.hdf5_file = h5py.File(self.hd5_path, mode='w')

        for pre in ['train', 'val']:
            if pre == 'train':
                shape = train_shape
                count = self.num_train
            else:
                shape = val_shape
                count = self.num_val
            self.hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8,
                                          maxshape=(None, shape[1], shape[2], shape[3]))
            self.hdf5_file.create_dataset("{}_round".format(pre), (count,), np.int16, maxshape=(None,))
            self.hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float, maxshape=(None,))
            for k, s in self.sets.items():
                self.hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count,), np.uint8, maxshape=(None,))

        self.process_index = 0
