import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, sides, BOX_PARAMETERS, BASE_TIME_STEP, PLAYER_WIDTH, PLAYER_HEIGHT
from annotator.utils import look_up_player_state
from annotator.api_requests import get_player_states, get_round_states
from annotator.game_values import HERO_SET, COLOR_SET, STATUS_SET, SPECTATOR_MODES


class PlayerStatusGenerator(DataGenerator):
    identifier = 'player_status'
    num_slots = 12
    num_variations = 1
    time_step = BASE_TIME_STEP
    secondary_time_step = round(BASE_TIME_STEP * 4, 1)
    use_window = False
    usable_annotations = ['M', 'O']

    def __init__(self):
        super(PlayerStatusGenerator, self).__init__()
        self.sets = {'hero': [na_lab] + HERO_SET,
                     'ult': [na_lab, 'no_ult', 'using_ult', 'has_ult'],
                     'alive': [na_lab, 'alive', 'dead'],
                     'side': [na_lab] + sides,
                     #'color': [na_lab] + COLOR_SET,
                     #'enemy_color': [na_lab] + COLOR_SET,
                     'spectator_mode': SPECTATOR_MODES,
                     #'switch': [na_lab, 'switch', 'not_switch'],
                     'status': ['normal', 'asleep', 'frozen', 'hacked', 'stunned', 'discord', 'resurrecting']
                     }
        for s in STATUS_SET:
            if s == 'status':
                continue
            if s in self.sets['status']:
                continue
            if not s:
                continue
            self.sets[s] = [na_lab] + ['not_' + s, s]
        self.save_set_info()
        self.image_width = PLAYER_WIDTH
        self.image_height =PLAYER_HEIGHT
        for s in sides:
            for i in range(6):
                self.slots.append((s, i))
        self.slot_params = {}
        self.check_set_info()
        self.has_status = False
        self.slot_indices = {}

    def figure_slot_params(self, r):
        film_format = r['stream_vod']['film_format']['code']
        left_params = BOX_PARAMETERS[film_format]['LEFT']
        right_params = BOX_PARAMETERS[film_format]['RIGHT']
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
        if 'ZOOMED_LEFT' in BOX_PARAMETERS[film_format]:
            self.zoomed_width = BOX_PARAMETERS[film_format]['ZOOMED_LEFT']['WIDTH']
            self.zoomed_height = BOX_PARAMETERS[film_format]['ZOOMED_LEFT']['HEIGHT']
            zoomed_left = BOX_PARAMETERS[film_format]['ZOOMED_LEFT']
            zoomed_right = BOX_PARAMETERS[film_format]['ZOOMED_RIGHT']
            self.zoomed_params = {}
            for side in sides:
                if side == 'left':
                    p = zoomed_left
                else:
                    p = zoomed_right
                for i in range(6):
                    self.zoomed_params[(side, i)] = {}
                    self.zoomed_params[(side, i)]['x'] = p['X'] + (p['WIDTH'] + p['MARGIN']) * i
                    self.zoomed_params[(side, i)]['y'] = p['Y']
        else:
            self.zoomed_params = self.slot_params

    def is_zoomed(self, time_point, side):
        for z in self.zooms[side]:
            if round(z['begin'] + 0.3, 1) <= time_point < round(z['end'] - 0.3, 1):
                return True
        return False

    def check_status(self, slot, time_point):
        for interval in self.status_state[slot]:
            if interval['end'] >= time_point >= interval['begin']:
                return True
        return False

    def ignore_time_point(self, time_point, side):
        for z in self.zooms[side]:
            if z['begin'] <= time_point < round(z['begin'] + 0.5, 1):
                return True
            if round(z['end'] - 0.5, 1) <= time_point <= z['end']:
                return True
        return False

    def get_data(self, r):
        self.status_state = {}
        self.left_color = r['game']['left_team']['color'].lower()
        self.right_color = r['game']['right_team']['color'].lower()
        self.left_color_hex = r['game']['left_team_color_hex'].lstrip('#')
        self.right_color_hex = r['game']['right_team_color_hex'].lstrip('#')
        self.left_color_hex = tuple(int(self.left_color_hex[i:i+2], 16) for i in (0, 2, 4))
        self.right_color_hex = tuple(int(self.right_color_hex[i:i+2], 16) for i in (0, 2, 4))
        self.spec_mode = r['spectator_mode'].lower()
        self.slot_indices = {}
        for s in self.slots:
            self.slot_indices[s] = {}
            for k in self.sets.keys():
                self.slot_indices[s][k] = 0
        self.states = get_player_states(r['id'])
        round_states = get_round_states(r['id'])
        self.zooms = {'left': [], 'right': []}
        zooms = round_states['zoomed_bars']
        for side in self.zooms.keys():
            for z in zooms[side]:
                if z['status'] == 'zoomed':
                    self.zooms[side].append(z)
        for slot in self.slots:
            print(slot)
            self.status_state[slot] = []
            intervals = self.states[slot[0]][str(slot[1])]['status']
            anti_intervals = [x for x in self.states[slot[0]][str(slot[1])]['antiheal'] if x['status'] == 'antiheal']
            nano_intervals = [x for x in self.states[slot[0]][str(slot[1])]['nanoboosted'] if x['status'] == 'nanoboosted']
            intervals = sorted(intervals+ anti_intervals+ nano_intervals, key=lambda x: x['begin'])
            if len(intervals) == 1:
                continue
            for interval in intervals:
                if interval['status'] == 'normal':
                    continue
                if not self.status_state[slot]:
                    self.status_state[slot].append({'begin': interval['begin'], 'end': interval['end']})
                else:
                    for existing_interval in self.status_state[slot]:
                        if existing_interval['end'] >= interval['begin'] >= existing_interval['begin']:
                            existing_interval['end'] = interval['end']
                            break
                        elif existing_interval['end'] >= interval['end'] >= existing_interval['begin']:
                            existing_interval['begin'] = interval['begin']
                            break
                    else:
                        self.status_state[slot].append({'begin': interval['begin'], 'end': interval['end']})

            print(self.status_state[slot])

    def display_current_frame(self, frame, time_point, frame_ind):
        for slot in self.slots:
            is_status = self.check_status(slot, time_point)
            if not is_status:
                if frame_ind % (self.secondary_time_step/self.time_step) != 0:
                    continue

            d = self.lookup_data(slot, time_point)
            print(slot, d)
            side = slot[0]
            zoomed = self.is_zoomed(time_point, side)
            if isinstance(slot, (list, tuple)):
                slot_name = '_'.join(map(str, slot))
            else:
                slot_name = slot
            if zoomed:
                params = self.zoomed_params[slot]
            else:
                params = self.slot_params[slot]
            x = params['x']
            y = params['y']
            if zoomed:
                box = frame[y: y + self.zoomed_height,
                      x: x + self.zoomed_width]
                box = cv2.resize(box, (self.image_height, self.image_width))
            else:
                box = frame[y: y + self.image_height,
                      x: x + self.image_width]
            cv2.imshow('{}_{}'.format(self.identifier, slot_name), box)

    def process_frame(self, frame_data, time_point, frame_ind):
        if not self.generate_data:
            return
        for slot in self.slots:
            side = slot[0]
            is_status = self.check_status(slot, time_point)
            if not is_status:
                if frame_ind % (self.secondary_time_step/self.time_step) != 0:
                    continue
            zoomed = self.is_zoomed(time_point, side)
            ignore = self.ignore_time_point(time_point, side)
            d = self.lookup_data(slot, time_point)
            if zoomed:
                params = self.zoomed_params[slot]
            else:
                params = self.slot_params[slot]
            x = params['x']
            y = params['y']
            variation_set = []
            while len(variation_set) < self.num_variations:
                x_offset = random.randint(-2, 2)
                y_offset = random.randint(-2, 2)
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
                if ignore:
                    self.ignored_indexes[pre].append(index)
                    self.process_index += 1
                    continue
                if zoomed:
                    if self.use_window:
                        prev = frame_data['prev'][y + y_offset: y + self.zoomed_height + y_offset,
                              x + x_offset: x + self.zoomed_width + x_offset]
                        next = frame_data['next'][y + y_offset: y + self.zoomed_height + y_offset,
                              x + x_offset: x + self.zoomed_width + x_offset]
                        prev = cv2.resize(prev, (self.image_height, self.image_width))
                        next = cv2.resize(next, (self.image_height, self.image_width))
                    box = frame_data['frame'][y + y_offset: y + self.zoomed_height + y_offset,
                          x + x_offset: x + self.zoomed_width + x_offset]
                    box = cv2.resize(box, (self.image_height, self.image_width))
                else:
                    if self.use_window:
                        prev = frame_data['prev'][y + y_offset: y + self.image_height + y_offset,
                              x + x_offset: x + self.image_width + x_offset]
                        next = frame_data['next'][y + y_offset: y + self.image_height + y_offset,
                              x + x_offset: x + self.image_width + x_offset]
                    box = frame_data['frame'][y + y_offset: y + self.image_height + y_offset,
                          x + x_offset: x + self.image_width + x_offset]
                if False:
                    cv2.imshow('frame_{}'.format(slot), box)
                    print(time_point)
                    print(d)
                    cv2.waitKey()
                box = np.transpose(box, axes=(2, 0, 1))
                if self.use_window:
                    prev = np.transpose(prev, axes=(2, 0, 1))
                    next = np.transpose(next, axes=(2, 0, 1))
                    self.data["{}_img".format(pre)][index, 0, ...] = prev[None]
                    self.data["{}_img".format(pre)][index, 1, ...] = box[None]
                    self.data["{}_img".format(pre)][index, 2, ...] = next[None]
                else:
                    self.data["{}_img".format(pre)][index, ...] = box[None]
                self.data["{}_round".format(pre)][index] = self.current_round_id

                self.data["{}_time_point".format(pre)][index] = time_point
                self.data['{}_color'.format(pre)][index, :] = d['color_hex']
                self.data['{}_enemy_color'.format(pre)][index, :] = d['enemy_color_hex']

                for k, s in self.sets.items():
                    self.data["{}_{}_label".format(pre, k)][index] = s.index(d[k])

                self.process_index += 1
                if self.debug:
                    filename = '{}_{}.jpg'.format(' '.join(d.values()), index).replace(':', '')
                    cv2.imwrite(os.path.join(self.training_directory, 'debug', pre,
                                         filename), np.transpose(box, axes=(1,2,0)))

    def add_new_round_info(self, r, reset=False):
        if r['id'] < 9359:
            self.generate_data = False
            return
        self.current_round_id = r['id']
        spec_mode_directory = os.path.join(self.training_directory, r['spectator_mode'].lower())
        os.makedirs(spec_mode_directory, exist_ok=True)
        self.hd5_path = os.path.join(spec_mode_directory, '{}.hdf5'.format(r['id']))
        if reset and os.path.exists(self.hd5_path):
            os.remove(self.hd5_path)
        if os.path.exists(self.hd5_path) or r['annotation_status'] not in self.usable_annotations:
            self.generate_data = False
            return
        self.time_step = BASE_TIME_STEP
        self.secondary_time_step = round(BASE_TIME_STEP * 4, 1)
        if r['annotation_status'] == 'O':
            self.time_step = round(self.time_step * 2, 1)
            self.secondary_time_step = round(self.secondary_time_step * 2, 1)
        self.has_status = False
        self.get_data(r)
        expected_duration = 0
        for beg, end in r['sequences']:
            beg += 0.1
            beg = round(beg, 1)
            end -= 0.1
            end = round(end, 1)
            if end <= beg:
                continue
            expected_duration += end - beg
        num_frames = 0
        for slot in self.slots:
            print(slot)
            status_duration = round(sum([x['end'] - x['begin'] for x in self.status_state[slot]]), 1)

            normal_duration = expected_duration - status_duration
            normal_frame_count = int(normal_duration / self.secondary_time_step)
            status_frame_count = int(status_duration / self.time_step)
            num_frames += normal_frame_count + status_frame_count
            print('NORMAL', normal_duration, normal_frame_count)
            print('STATUS', status_duration, status_frame_count)
            if not self.has_status:
                self.has_status = status_duration > 0
        num_frames *= self.num_variations
        print('TOTAL FRAMES', num_frames)
        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train
        self.analyzed_rounds.append(r['id'])
        self.current_round_id = r['id']
        self.generate_data = True
        self.figure_slot_params(r)

        self.indexes = random.sample(range(num_frames), num_frames)

        if self.use_window:
            train_shape = (self.num_train, 3, 3, int(self.image_height * self.resize_factor), int(self.image_width * self.resize_factor))
            val_shape = (self.num_val, 3, 3, int(self.image_height * self.resize_factor), int(self.image_width * self.resize_factor))
        else:
            train_shape = (self.num_train, 3, int(self.image_height * self.resize_factor), int(self.image_width * self.resize_factor))
            val_shape = (self.num_val, 3, int(self.image_height * self.resize_factor), int(self.image_width * self.resize_factor))
        self.data = {}
        self.ignored_indexes = {}
        for pre in ['train', 'val']:
            self.ignored_indexes[pre] = []
            if pre == 'train':
                shape = train_shape
                count = self.num_train
            else:
                shape = val_shape
                count = self.num_val
            self.data["{}_img".format(pre)] = np.zeros(shape, dtype=np.uint8)
            self.data['{}_color'.format(pre)] = np.zeros((count,3), dtype=np.uint8)
            self.data['{}_enemy_color'.format(pre)] = np.zeros((count,3), dtype=np.uint8)
            self.data["{}_round".format(pre)] = np.zeros((count,), dtype=np.int16)
            self.data["{}_time_point".format(pre)] = np.zeros((count,), dtype=np.float)
            for k, s in self.sets.items():
                self.data["{}_{}_label".format(pre, k)] =  np.zeros((count,), dtype=np.uint8)

        self.process_index = 0

    def lookup_data(self, slot, time_point):
        side, index = slot
        states = self.states[side][str(index)]
        d = {}
        for k in self.sets.keys():
            if k in ['spectator_mode', 'side', 'color', 'enemy_color']:
                continue
            if len(states[k]) == 0:
                for k, v in self.sets.items():
                    d[k] = v[0] # n/a
                break
            while time_point >= states[k][self.slot_indices[slot][k]]['end'] and self.slot_indices[slot][k] < len(states[k]) - 1:
                self.slot_indices[slot][k] += 1
            index = self.slot_indices[slot][k]
            if k == 'hero':
                d[k] = states[k][index]['hero']['name'].lower()
            else:
                d[k] = states[k][index]['status'].lower()
        if d['hero'] == na_lab:
            d['ult'] = na_lab
            d['antiheal'] = na_lab
            d['immortal'] = na_lab
            d['alive'] = na_lab
        #d = look_up_player_state(slot[0], slot[1], time_point, self.states, self.has_status)
        d['spectator_mode'] = self.spec_mode
        if slot[0] == 'left':
            d['color'] = self.left_color
            d['enemy_color'] = self.right_color
            d['color_hex'] = self.left_color_hex
            d['enemy_color_hex'] = self.right_color_hex
        else:
            d['color'] = self.right_color
            d['enemy_color'] = self.left_color
            d['color_hex'] = self.right_color_hex
            d['enemy_color_hex'] = self.left_color_hex
        d['side'] = slot[0]
        return d

    def cleanup_round(self):
        if not self.generate_data:
            return
        for pre, ignored in self.ignored_indexes.items():
            if ignored:
                for k, v in self.data.items():
                    self.data[k] = np.delete(v, ignored, axis=0)
        super(PlayerStatusGenerator, self).cleanup_round()