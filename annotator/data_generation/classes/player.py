import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab, sides, BOX_PARAMETERS
from annotator.utils import look_up_player_state
from annotator.api_requests import get_player_states, get_round_states
from annotator.game_values import HERO_SET, COLOR_SET, STATUS_SET, SPECTATOR_MODES


class PlayerStatusGenerator(DataGenerator):
    identifier = 'player_status'
    num_slots = 12
    num_variations = 1
    time_step = 0.1
    secondary_time_step = 0.5

    def __init__(self):
        super(PlayerStatusGenerator, self).__init__()
        self.sets = {'hero': [na_lab] + HERO_SET,
                     'ult': [na_lab, 'no_ult', 'using_ult', 'has_ult'],
                     'alive': [na_lab, 'alive', 'dead'],
                     'side': [na_lab] + sides,
                     'color': [na_lab] + COLOR_SET,
                     'spectator_mode': SPECTATOR_MODES,
                     'switch': [na_lab, 'switch', 'not_switch'],
                     'status': ['normal', 'asleep', 'frozen', 'hacked', 'stunned']
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
        params = BOX_PARAMETERS['O']['LEFT']
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        for s in sides:
            for i in range(6):
                self.slots.append((s, i))
        self.slot_params = {}
        self.check_set_info()
        self.has_status = False

    def figure_slot_params(self, r):
        if r['stream_vod']['film_format'] != 'O':
            film_format = r['stream_vod']['film_format']
        else:
            film_format = r['game']['match']['event']['film_format']
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
            if z['begin'] + 0.3 <= time_point < z['end'] - 0.3:
                return True
        return False

    def get_data(self, r):
        self.status_state = {}
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
            for s in STATUS_SET:
                if not s:
                    continue
                intervals = self.states[slot[0]][str(slot[1])][s]
                if len(intervals) == 1:
                    continue
                for interval in intervals:
                    if interval['status'].startswith('not_'):
                        continue
                    if not self.status_state[slot]:
                        self.status_state[slot].append({'begin':interval['begin'], 'end':interval['end']})
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

                    print(interval)
            print(self.status_state[slot])

    def check_status(self, slot, time_point):
        for interval in self.status_state[slot]:
            if interval['end'] >= time_point >= interval['begin']:
                return True
        return False

    @property
    def minimum_time_step(self):
        if self.has_status:
            return self.time_step
        else:
            return self.secondary_time_step

    def display_current_frame(self, frame, time_point):
        for slot in self.slots:
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
                box = cv2.resize(box, (64, 64))
            else:
                box = frame[y: y + self.image_height,
                      x: x + self.image_width]
            cv2.imshow('{}_{}'.format(self.identifier, slot_name), box)

    def process_frame(self, frame, time_point, frame_ind):
        if not self.generate_data:
            return
        for slot in self.slots:
            if self.has_status:
                is_status = self.check_status(slot, time_point)
            else:
                is_status = False
            if not is_status:
                if frame_ind % (self.secondary_time_step/self.minimum_time_step) != 0:
                    continue
            side = slot[0]
            zoomed = self.is_zoomed(time_point, side)
            #if not zoomed: # FIXME
            #   return
            d = self.lookup_data(slot, time_point)
            if zoomed:
                params = self.zoomed_params[slot]
            else:
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
                if zoomed:
                    box = frame[y + y_offset: y + self.zoomed_height + y_offset,
                          x + x_offset: x + self.zoomed_width + x_offset]
                    box = cv2.resize(box, (self.image_height, self.image_width))
                else:
                    box = frame[y + y_offset: y + self.image_height + y_offset,
                          x + x_offset: x + self.image_width + x_offset]
                if False:
                    cv2.imshow('frame_{}'.format(slot), box)
                    print(time_point)
                    print(d)
                    cv2.waitKey()
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
        if os.path.exists(self.hd5_path) or r['annotation_status'] not in self.usable_annotations:
            self.generate_data = False
            return
        self.get_data(r)

        num_frames = 0
        expected_duration = 0
        for beg, end in r['sequences']:
            expected_duration += end - beg
        self.has_status = False
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
        print('TOTAL FRAMES', num_frames)
        num_frames *= self.num_variations
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

        self.left_color = r['game']['left_team']['color'].lower()
        self.right_color = r['game']['right_team']['color'].lower()
        self.spec_mode = r['spectator_mode'].lower()

    def lookup_data(self, slot, time_point):
        d = look_up_player_state(slot[0], slot[1], time_point, self.states, self.has_status)
        d['spectator_mode'] = self.spec_mode
        if slot[0] == 'left':
            d['color'] = self.left_color
        else:
            d['color'] = self.right_color
        d['side'] = slot[0]
        return d
