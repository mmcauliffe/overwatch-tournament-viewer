import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, BOX_PARAMETERS


class GameGenerator(DataGenerator):
    identifier = 'game'
    resize_factor = 0.5
    num_variations = 2
    time_step = 1

    def __init__(self):
        super(GameGenerator, self).__init__()
        self.immutable_sets = {}
        self.sets = {
            'game': ['not_game', 'game'],
        }
        params = BOX_PARAMETERS['O']['MID']
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        self.check_set_info()
        self.slots = [1]
        self.save_set_info()

    def figure_slot_params(self, r):
        params = BOX_PARAMETERS[r['film_format']]['MID']
        self.slot_params = {}
        self.slot_params[1] = {'x': params['X'], 'y': params['Y']}

    def lookup_data(self, slot, time_point):
        d = {'game': 'not_game'}
        for s in self.sequences:
            if s[0] <= time_point <= s[1]:
                d['game'] = 'game'
                break
        return d

    def add_new_round_info(self, r):
        super(GameGenerator, self).add_new_round_info(r)
        if not self.generate_data:
            return
        self.sequences = r['sequences']
