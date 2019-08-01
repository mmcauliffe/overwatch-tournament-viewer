import h5py
import os
import random
import cv2
import heapq
import numpy as np
import lmdb
import pickle

from annotator.data_generation.classes.ctc import CTCDataGenerator, write_cache
from annotator.game_values import COLOR_SET, HERO_SET, LABEL_SET
from annotator.config import BOX_PARAMETERS, na_lab
from annotator.api_requests import get_kf_events
from annotator.utils import get_event_ranges


def construct_kf_at_time(events, time):
    window = 7.3
    possible_kf = []
    k = 6
    for e in events:
        if e['time_point'] > time + 0.25:
            break
        elif e['time_point'] > time:
            possible_kf.append({'time_point': e['time_point'],
                                'first_hero': 'n/a', 'first_color': 'n/a', 'ability': 'n/a', 'headshot': 'n/a',
                                'second_hero': 'n/a',
                                'second_color': 'n/a', })
        if time - window <= e['time_point'] <= time:
            possible_kf.append(e)
    possible_kf = possible_kf[-6:]
    possible_kf.reverse()
    return possible_kf


class KillFeedCTCGenerator(CTCDataGenerator):
    identifier = 'kill_feed_ctc'
    num_slots = 6
    num_variations = 1
    data_bytes = 24200
    time_step = 0.5

    def __init__(self, debug=False):
        super(KillFeedCTCGenerator, self).__init__(debug=debug, map_size=50995116277)
        self.label_set = LABEL_SET
        self.save_label_set()
        self.image_width = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['WIDTH']
        self.image_height = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['HEIGHT']
        self.debug=debug
        if self.debug:
            os.makedirs(os.path.join(self.training_directory, 'debug', 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.training_directory, 'debug', 'val'), exist_ok=True)
        self.slots = range(6)
        self.current_round_id = None
        self.check_set_info()
        self.half_size_npcs = False

    def figure_slot_params(self, r):
        self.slot_params = {}
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['KILL_FEED_SLOT']
        self.half_size_npcs = r['spectator_mode'] == 'Status'
        for s in self.slots:
            self.slot_params[s] = {}
            self.slot_params[s]['x'] = params['X']
            self.slot_params[s]['y'] = params['Y'] + (params['HEIGHT'] + params['MARGIN']) * (s)

    def lookup_data(self, slot_data, time_point):
        raw_sequence = []
        d = slot_data
        if d['headshot'] and not d['ability'].endswith('headshot'):
            d['ability'] += ' headshot'
        if d['first_player'] == 'n/a':
            pass
        else:
            raw_sequence.append(d['first_color'])
            raw_sequence.append(d['first_hero'])
            if d['assisting_heroes']:
                for h in d['assisting_heroes']:
                    raw_sequence.append(h.lower() + '_assist')

        raw_sequence.append(d['ability'])
        second = d['second_hero']
        if second not in HERO_SET and second+'_npc' in LABEL_SET:
            second += '_npc'
        raw_sequence.append(second)
        raw_sequence.append(d['second_color'])
        raw_sequence = [x for x in raw_sequence]
        return raw_sequence

    def display_current_frame(self, frame, time_point):
        shift = 0
        kf = construct_kf_at_time(self.states, time_point)
        for slot in self.slots:
            if isinstance(slot, (list, tuple)):
                slot_name = '_'.join(map(str, slot))
            else:
                slot_name = slot
            params = self.slot_params[slot]
            if slot > len(kf) - 1 or kf[slot]['second_hero'] == na_lab:
                sequence = []
                is_npc = False
            else:
                is_npc = kf[slot]['second_hero'] not in HERO_SET + ['b.o.b._npc']
            x = params['x']
            y = params['y']
            #if self.half_size_npcs and is_npc:
            #    y -= int(self.image_height /4) - 2
            box = frame[y - shift: y + self.image_height - shift,
                  x: x + self.image_width]
            cv2.imshow('{}_{}'.format(self.identifier, slot_name), box)
            if self.half_size_npcs and is_npc:
                shift += int(self.image_height /4) + 5

    def process_frame(self, frame, time_point):
        if not self.generate_data:
            return
        for rd in self.ranges:
            if rd['begin'] <= time_point <= rd['end']:
                break
        else:
            return

        kf = construct_kf_at_time(self.states, time_point)
        if not kf:
            return
        shift = 0
        for s in self.slots:
            params = self.slot_params[s]
            if s > len(kf) - 1 or kf[s]['second_hero'] == na_lab:
                raw_sequence = []
                is_npc = False
            else:
                raw_sequence = self.lookup_data(kf[s], time_point)
                is_npc = kf[s]['second_hero'] not in HERO_SET + ['b.o.b._npc']
            if not raw_sequence:
                continue
            variation_set = []
            while len(variation_set) < self.num_variations:
                x_offset = random.randint(-4, 4)
                y_offset = random.randint(-4, 4)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))

            x = params['x']
            y = params['y']

            #if self.half_size_npcs and is_npc:
            #    y -= int(self.image_height / 4) - 2

            for i, (x_offset, y_offset) in enumerate(variation_set):

                box = frame[y + y_offset - shift: y + self.image_height + y_offset - shift,
                      x + x_offset: x + self.image_width + x_offset]
                box = np.transpose(box, axes=(2, 0, 1))
                train_check = random.random() <= 0.8
                if train_check:
                    pre = 'train'
                    index = self.previous_train_count
                    image_key = 'image-%09d' % index
                    label_key = 'label-%09d' % index
                    round_key = 'round-%09d' % index
                    time_point_key = 'time_point-%09d' % index
                    self.train_cache[image_key] = pickle.dumps(box)
                    self.train_cache[label_key] = ','.join(raw_sequence)
                    self.train_cache[round_key] = str(self.current_round_id)
                    self.train_cache[time_point_key] = '{:.1f}'.format(time_point)
                    if len(self.train_cache) >= 4000:
                        write_cache(self.train_env, self.train_cache)
                        self.train_cache = {}
                    self.previous_train_count += 1
                else:
                    pre = 'val'
                    index = self.previous_val_count
                    image_key = 'image-%09d' % index
                    label_key = 'label-%09d' % index
                    round_key = 'round-%09d' % index
                    time_point_key = 'time_point-%09d' % index
                    self.val_cache[image_key] = pickle.dumps(box)
                    self.val_cache[label_key] = ','.join(raw_sequence)
                    self.val_cache[round_key] = str(self.current_round_id)
                    self.val_cache[time_point_key] = '{:.1f}'.format(time_point)
                    if len(self.val_cache) >= 4000:
                        write_cache(self.val_env, self.val_cache)
                        self.val_cache = {}
                    self.previous_val_count += 1
                if self.debug:
                    filename = '{}_{}.jpg'.format(' '.join(raw_sequence), self.process_index)
                    cv2.imwrite(os.path.join(self.training_directory, 'debug', pre,
                                         filename), box)

                self.process_index += 1
            if self.half_size_npcs and is_npc:
                shift += int(self.image_height /4) + 5
        if time_point % 5 == 0:
            self.states = [x for x in self.states if x['time_point'] > time_point - 8]

    def get_data(self, r):
        self.states = get_kf_events(r['id'])

    def add_new_round_info(self, r):
        super(KillFeedCTCGenerator, self).add_new_round_info(r)
        if not self.generate_data:
            return
        self.ranges = get_event_ranges(self.states, r['end'] - r['begin'])

    def calculate_map_size(self, rounds):
        num_frames = 0
        for r in rounds:
            ranges = get_event_ranges(get_kf_events(r['id']), r['end'] - r['begin'])
            for rd in ranges:
                num_frames +=int((rd['end'] - rd['begin']) / self.time_step)
        overall = num_frames * self.data_bytes * 10 * self.num_variations * self.num_slots
        self.train_map_size = int(overall * 0.82)
        self.val_map_size = int(overall * 0.22)
        print(self.train_map_size)
