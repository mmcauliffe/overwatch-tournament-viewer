import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, BOX_PARAMETERS
from annotator.utils import look_up_round_state
from annotator.api_requests import get_round_states
from annotator.game_values import COLOR_SET, MAP_SET, MAP_MODE_SET, SPECTATOR_MODES


class MidStatusGenerator(DataGenerator):
    identifier = 'mid'
    resize_factor = 0.5
    num_variations = 4
    time_step = 1
    data_bytes = 21500

    def __init__(self, debug=False):
        super(MidStatusGenerator, self).__init__(debug)
        self.sets = {
            'overtime': [na_lab] + ['not_overtime', 'overtime'],
            'point_status': [na_lab] + sorted(['Assault_1', 'Assault_2',
                                               'Hybrid_1', 'Hybrid_2', 'Hybrid_3',
                                               'Escort_1', 'Escort_2', 'Escort_3'] +
                                              ['Control_' + x for x in ['n/a'] + COLOR_SET]),
            'attacking_color': [na_lab] + COLOR_SET,
            'map': [na_lab] + MAP_SET,
            'map_mode': [na_lab] + MAP_MODE_SET,
            'round_number': [str(x) for x in range(1, 10)],
            'spectator_mode': SPECTATOR_MODES
        }
        params = BOX_PARAMETERS['O']['MID']
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        self.check_set_info()
        self.slots = [1]
        self.save_set_info()

    def figure_slot_params(self, r):
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['MID']
        self.slot_params = {}
        self.slot_params[1] = {'x': params['X'], 'y': params['Y']}

    def lookup_data(self, slot, time_point):
        d = look_up_round_state(time_point, self.states)
        for k, v in self.sets.items():
            if k == 'point_status':
                if d[k] == 'Control_none':
                    d[k] = 'Control_n/a'
                elif d[k] == 'Control_left':
                    d[k] = 'Control_' + self.left_color
                elif d[k] == 'Control_right':
                    d[k] = 'Control_' + self.right_color

        d['attacking_color'] = self.attacking_color
        d['map'] = self.map
        d['spectator_mode'] = self.spec_mode
        d['map_mode'] = self.map_mode
        d['round_number'] = self.round_number
        return d

    def add_new_round_info(self, r):
        super(MidStatusGenerator, self).add_new_round_info(r)
        if not self.generate_data:
            return
        self.attacking_color = r['attacking_color'].lower()
        self.map = r['game']['map']['name'].lower()
        self.map_mode = r['game']['map']['mode'].lower()
        self.spec_mode = r['spectator_mode'].lower()
        self.round_number = str(r['round_number'])
        self.left_color = r['game']['left_team']['color'].lower()
        self.right_color = r['game']['right_team']['color'].lower()
        self.states = get_round_states(r['id'])

    def calculate_map_size(self, rounds):
        num_frames = 0
        for r in rounds:
            for beg, end in r['sequences']:
                num_frames += int((end - beg) / self.time_step)
        overall = num_frames * self.data_bytes * 10 * self.num_variations * self.num_slots
        self.train_map_size = int(overall * 0.82)
        self.val_map_size = int(overall * 0.22)
        print(self.train_map_size)
