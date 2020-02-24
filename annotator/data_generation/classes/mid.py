import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, BOX_PARAMETERS, BASE_TIME_STEP
from annotator.utils import look_up_round_state
from annotator.api_requests import get_round_states
from annotator.game_values import COLOR_SET, MAP_SET, MAP_MODE_SET, SPECTATOR_MODES


class MidStatusGenerator(DataGenerator):
    identifier = 'mid'
    resize_factor = 0.5
    num_variations = 4
    time_step = round(BASE_TIME_STEP * 5, 1)

    def __init__(self, debug=False):
        super(MidStatusGenerator, self).__init__(debug=debug)
        self.sets = {
            'overtime': [na_lab] + ['not_overtime', 'overtime'],
            'point_status': [na_lab] + sorted(['Assault_1', 'Assault_2',
                                               'Hybrid_1', 'Hybrid_2', 'Hybrid_3',
                                               'Escort_1', 'Escort_2', 'Escort_3'] +
                                              ['Control_' + x for x in ['neither', 'left', 'right']]),
            'attacking_side': ['neither', 'left', 'right'],
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
        film_format = r['stream_vod']['film_format']['code']
        params = BOX_PARAMETERS[film_format]['MID']
        self.slot_params = {}
        self.slot_params[1] = {'x': params['X'], 'y': params['Y']}

    def lookup_data(self, slot, time_point):
        d = look_up_round_state(time_point, self.states)

        d['attacking_side'] = self.attacking_side
        d['map'] = self.map
        d['spectator_mode'] = self.spectator_mode
        d['map_mode'] = self.map_mode
        d['round_number'] = self.round_number
        return d

    def add_new_round_info(self, r, reset=False):
        super(MidStatusGenerator, self).add_new_round_info(r, reset=reset)
        if not self.generate_data:
            return

        self.states = get_round_states(r['id'])
        self.left_color = r['game']['left_team']['color'].lower()
        self.attacking_side = r['attacking_side'].lower()
        if self.attacking_side == 'l':
            self.attacking_side = 'left'
        elif self.attacking_side == 'r':
            self.attacking_side = 'right'
        else:
            self.attacking_side = 'neither'
        self.right_color = r['game']['right_team']['color'].lower()
        self.map = r['game']['map']['name'].lower()
        self.map_mode = r['game']['map']['mode'].lower()
        self.round_number = str(r['round_number'])
        self.spectator_mode = r['spectator_mode'].lower()
