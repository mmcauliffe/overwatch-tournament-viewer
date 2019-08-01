import h5py
import os
import random
import cv2
import numpy as np
import sys

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, BOX_PARAMETERS
from annotator.utils import look_up_round_state
from annotator.api_requests import get_round_states
import pickle


class ReplayStatusGenerator(DataGenerator):
    identifier = 'replay'
    resize_factor = 0.5
    num_variations = 2
    time_step = 1

    def __init__(self, debug=False):
        super(ReplayStatusGenerator, self).__init__(debug=debug)
        self.immutable_sets = {}
        self.sets = {
            'replay': ['not_replay', 'replay'],
        }
        params = BOX_PARAMETERS['O']['REPLAY']
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        self.check_set_info()
        self.slots = [1]
        self.save_set_info()

    def figure_slot_params(self, r):
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['REPLAY']
        self.slot_params = {}
        self.slot_params[1] = {'x': params['X'], 'y': params['Y']}

    def lookup_data(self, slot, time_point):
        d = look_up_round_state(time_point, self.states)
        return d

    def add_new_round_info(self, r):
        self.current_round_id = r['id']
        self.hd5_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        if os.path.exists(self.hd5_path):
            self.generate_data = False
            return
        self.generate_data = True
        self.states = get_round_states(r['id'])
        for s in self.states['replays']:
            if s['status'] == 'replay':
                break
        else:
            self.generate_data = False
        if not self.generate_data:
            return
        self.get_data(r)

        num_frames = int((r['end'] - r['begin']) / self.time_step) + 1
        for beg, end in r['sequences']:
            expected_duration = end - beg
            expected_frame_count = expected_duration / self.time_step
            num_frames += (int(expected_frame_count) + 1)

        num_frames *= self.num_variations * self.num_slots
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
