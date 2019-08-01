import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.sequence import SequenceDataGenerator
from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, sides, BOX_PARAMETERS
from annotator.utils import look_up_player_state
from annotator.api_requests import get_player_states
from annotator.game_values import HERO_SET, COLOR_SET, STATUS_SET, SPECTATOR_MODES


class PlayerStatusGenerator(DataGenerator):
    identifier = 'player_status'
    num_slots = 12
    num_variations = 1
    data_bytes = 13000
    time_step = 0.5

    def __init__(self,debug=False):
        super(PlayerStatusGenerator, self).__init__(debug, map_size=509951162777)
        self.sets = {'hero': [na_lab] + HERO_SET,
                     'ult': [na_lab, 'no_ult', 'using_ult', 'has_ult'],
                     'alive': [na_lab, 'alive', 'dead'],
            'side': [na_lab] + sides,
            'color': [na_lab] + COLOR_SET, 'spectator_mode': SPECTATOR_MODES}
        for s in STATUS_SET:
            if not s:
                continue
            self.sets[s] = [na_lab] + ['not_' + s, s]
        self.save_set_info()
        params = BOX_PARAMETERS['O']['LEFT']
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        for s in sides:
            for i in range(6):
                self.slots.append((s, i))
        self.slot_params = {}
        self.check_set_info()

    def figure_slot_params(self, r):
        left_params = BOX_PARAMETERS[r['stream_vod']['film_format']]['LEFT']
        right_params = BOX_PARAMETERS[r['stream_vod']['film_format']]['RIGHT']
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

    def add_new_round_info(self, r):
        super(PlayerStatusGenerator, self).add_new_round_info(r)
        if not self.generate_data:
            return
        self.spec_mode = r['spectator_mode'].lower()

        self.states = get_player_states(r['id'])
        self.left_color = r['game']['left_team']['color'].lower()
        self.right_color = r['game']['right_team']['color'].lower()

    def lookup_data(self, slot, time_point):
        d = look_up_player_state(slot[0], slot[1], time_point, self.states)
        d['spectator_mode'] = self.spec_mode
        if slot[0] == 'left':
            d['color'] = self.left_color
        else:
            d['color'] = self.right_color
        d['side'] = slot[0]
        return d

    def calculate_map_size(self, rounds):
        num_frames = 0
        for r in rounds:
            for beg, end in r['sequences']:
                num_frames += int((end-beg) / self.time_step)

        overall = num_frames * self.data_bytes * 10 * self.num_variations * self.num_slots
        self.train_map_size = int(overall * 0.82)
        self.val_map_size = int(overall * 0.22)
        print(self.train_map_size)
