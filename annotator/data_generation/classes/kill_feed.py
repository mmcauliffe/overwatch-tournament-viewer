import h5py
import os
import random
import cv2
import heapq
import numpy as np

from annotator.data_generation.classes.ctc import CTCDataGenerator
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
    num_variations = 2

    def __init__(self):
        super(KillFeedCTCGenerator, self).__init__()
        self.label_set = LABEL_SET
        self.save_label_set()
        self.image_width = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['WIDTH']
        self.image_height = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['HEIGHT']
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
        sequence = []
        d = slot_data
        if d['headshot'] and not d['ability'].endswith('headshot'):
            d['ability'] += ' headshot'
        if d['first_player'] == 'n/a':
            pass
        else:
            sequence.append(self.label_set.index(d['first_color']))
            sequence.append(self.label_set.index(d['first_hero']))
            if d['assisting_heroes']:
                for h in d['assisting_heroes']:
                    sequence.append(self.label_set.index(h.lower() + '_assist'))

        sequence.append(self.label_set.index(d['ability']))
        second = d['second_hero']
        if second not in HERO_SET and second+'_npc' in LABEL_SET:
            second += '_npc'
        sequence.append(self.label_set.index(second))
        sequence.append(self.label_set.index(d['second_color']))
        return sequence

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
        shift = 0
        for s in self.slots:
            params = self.slot_params[s]
            if s > len(kf) - 1 or kf[s]['second_hero'] == na_lab:
                sequence = []
                is_npc = False
            else:
                sequence = self.lookup_data(kf[s], time_point)
                is_npc = kf[s]['second_hero'] not in HERO_SET + ['b.o.b._npc']

            variation_set = [(0, 0)]
            while len(variation_set) < self.num_variations:
                x_offset = random.randint(-5, 5)
                y_offset = random.randint(-5, 5)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))

            x = params['x']
            y = params['y']

            #if self.half_size_npcs and is_npc:
            #    y -= int(self.image_height / 4) - 2

            for i, (x_offset, y_offset) in enumerate(variation_set):
                if self.process_index > len(self.indexes) - 1:
                    continue
                index = self.indexes[self.process_index]
                if index < self.num_train:
                    pre = 'train'
                else:
                    pre = 'val'
                    index -= self.num_train
                box = frame[y + y_offset - shift: y + self.image_height + y_offset - shift,
                      x + x_offset: x + self.image_width + x_offset]
                box = np.swapaxes(box, 1, 0)
                self.hdf5_file["{}_img".format(pre)][index, ...] = box[None]
                self.hdf5_file["{}_round".format(pre)][index] = self.current_round_id
                # self.train_mean += box / self.hdf5_file['train_img'].shape[0]
                sequence_length = len(sequence)
                if sequence:
                    self.hdf5_file["{}_label_sequence_length".format(pre)][index] = sequence_length
                    self.hdf5_file["{}_label_sequence".format(pre)][index, 0:len(sequence)] = sequence
                self.process_index += 1
            if self.half_size_npcs and is_npc:
                shift += int(self.image_height /4) + 5
        if time_point % 5 == 0:
            self.states = [x for x in self.states if x['time_point'] > time_point - 8]

    def get_data(self, r):
        self.states = get_kf_events(r['id'])

    def add_new_round_info(self, r):
        self.current_round_id = r['id']
        self.hd5_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        if os.path.exists(self.hd5_path):
            self.generate_data = False
            return
        self.get_data(r)
        self.ranges = get_event_ranges(self.states, r['end'] - r['begin'])
        num_frames = 0
        for rd in self.ranges:
            expected_duration = rd['end'] - rd['begin']
            expected_frame_count = expected_duration / self.time_step
            num_frames += (int(expected_frame_count) + 1) * self.num_slots

        num_frames *= self.num_variations
        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train
        self.analyzed_rounds.append(r['id'])
        self.current_round_id = r['id']
        self.generate_data = True
        self.figure_slot_params(r)

        self.indexes = random.sample(range(num_frames), num_frames)

        train_shape = (self.num_train, self.image_width, self.image_height, 3)
        val_shape = (self.num_val, self.image_width, self.image_height, 3)
        self.hdf5_file = h5py.File(self.hd5_path, mode='w')
        # self.train_mean = np.zeros(train_shape[1:], np.float32)
        # self.hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
        for pre in ['train', 'val']:
            if pre == 'train':
                shape = train_shape
                count = self.num_train
            else:
                shape = val_shape
                count = self.num_val
            self.hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8,
                                          maxshape=(None, shape[1], shape[2], shape[3]))
            self.hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32, maxshape=(None,))
            self.hdf5_file.create_dataset("{}_label_sequence".format(pre), (count, self.max_sequence_length),
                                          np.uint32, maxshape=(None, self.max_sequence_length),
                                          fillvalue=len(self.label_set))
            self.hdf5_file.create_dataset("{}_label_sequence_length".format(pre), (count,), np.uint8,
                                          maxshape=(None,), fillvalue=1)

        self.process_index = 0
