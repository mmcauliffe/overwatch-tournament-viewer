import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, BOX_PARAMETERS
from annotator.api_requests import get_game_states
from annotator.utils import look_up_game_state


class GameGenerator(DataGenerator):
    identifier = 'game'
    resize_factor = 0.2
    time_step = 0.5

    def __init__(self):
        super(GameGenerator, self).__init__()
        self.immutable_sets = {}
        self.sets = {
            'game': ['not_game', 'game', 'pause', 'replay','smaller_window'],
            'left': ['n/a', 'not_zoom', 'zoom'],
            'right': ['n/a', 'not_zoom', 'zoom'],
        }

        self.image_width = 1280
        self.image_height = 720
        self.check_set_info()
        self.slots = [1]
        self.save_set_info()

    def figure_slot_params(self, r):
        self.slot_params = {}
        self.slot_params[1] = {'x': 0, 'y': 0}

    def lookup_data(self, slot, time_point):
        d = look_up_game_state(time_point, self.states)
        return d

    def process_frame(self, frame, time_point, frame_ind):
        if not self.generate_data:
            return
        for slot in self.slots:
            d = self.lookup_data(slot, time_point)
            params = self.slot_params[slot]
            x = params['x']
            y = params['y']
            if self.process_index > len(self.indexes) - 1:
                continue
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
        super(GameGenerator, self).add_new_round_info(r)
        if not self.generate_data:
            return
        self.states = get_game_states(r['id'])
