import h5py
import os
import random
import cv2
import heapq
import numpy as np

from annotator.data_generation.classes.ctc import CTCDataGenerator
from annotator.game_values import COLOR_SET, HERO_SET, LABEL_SET, SPECTATOR_MODES
from annotator.config import BOX_PARAMETERS, na_lab, BASE_TIME_STEP
from annotator.api_requests import get_kf_events
from annotator.utils import get_event_ranges


def check_is_npc(event):
    return event['second_hero'] not in HERO_SET + ['b.o.b._npc']


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
                                'second_color': 'n/a', 'is_npc': check_is_npc(e)})
        if time - window <= e['time_point'] <= time:
            e['is_npc'] = check_is_npc(e)
            possible_kf.append(e)
    possible_kf = possible_kf[-6:]
    possible_kf.reverse()
    return possible_kf


class KillFeedCTCGenerator(CTCDataGenerator):
    identifier = 'kill_feed_ctc'
    num_slots = 6
    num_variations = 1
    time_step = round(BASE_TIME_STEP * 2, 1)

    def __init__(self, debug=False):
        super(KillFeedCTCGenerator, self).__init__()
        self.label_set = LABEL_SET
        self.exist_label_set = ['empty', 'full_sized', 'half_sized']
        self.save_label_set()
        self.image_width = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['WIDTH']
        self.image_height = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['HEIGHT']
        self.margin = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['MARGIN']
        self.debug=debug
        if self.debug:
            os.makedirs(os.path.join(self.training_directory, 'debug', 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.training_directory, 'debug', 'val'), exist_ok=True)
        self.slots = range(6)
        self.current_round_id = None
        self.check_set_info()
        self.half_size_npcs = False

    def save_label_set(self):
        super(KillFeedCTCGenerator, self).save_label_set()
        path = os.path.join(self.training_directory, 'exist_label_set.txt')
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf8') as f:
                for c in self.exist_label_set:
                    f.write('{}\n'.format(c))

    def figure_slot_params(self, r):
        from datetime import datetime
        self.slot_params = {}
        film_format = r['stream_vod']['film_format']
        params = BOX_PARAMETERS[film_format]['KILL_FEED_SLOT']
        broadcast_date = datetime.strptime(r['stream_vod']['broadcast_date'], '%Y-%m-%dT%H:%M:%SZ')
        status_date_start = datetime(2019, 1, 1)
        self.half_size_npcs = broadcast_date > status_date_start
        self.margin = params['MARGIN']
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
            first_color = d['first_color']
            #if self.spec_mode != 'original':
            #    if first_color != 'white':
            #        first_color = 'nonwhite'
            raw_sequence.append(first_color)
            raw_sequence.append(d['first_hero'])
            if d['assisting_heroes']:
                for h in d['assisting_heroes']:
                    raw_sequence.append(h.lower() + '_assist')
        if d['ability'] == 'n/a':
            d['ability'] = 'primary'
        raw_sequence.append(d['ability'])
        second = d['second_hero']
        if second not in HERO_SET and second+'_npc' in LABEL_SET:
            second += '_npc'
        raw_sequence.append(second)
        second_color = d['second_color']
        #if self.spec_mode != 'original':
        #    if second_color != 'white':
        #        second_color = 'nonwhite'
        raw_sequence.append(second_color)
        raw_sequence = [x for x in raw_sequence]
        sequence = [self.label_set.index(x) for x in raw_sequence]
        return sequence, raw_sequence

    def display_current_frame(self, frame, time_point):
        shift = 0
        kf = construct_kf_at_time(self.states, time_point)
        show = False
        for slot in self.slots:
            if isinstance(slot, (list, tuple)):
                slot_name = '_'.join(map(str, slot))
            else:
                slot_name = slot
            params = self.slot_params[slot]
            if slot > len(kf) - 1:
                is_npc = False
            else:
                is_npc = kf[slot]['is_npc']
                sequence, raw_sequence = self.lookup_data(kf[slot], time_point)
                print(sequence)
                print(raw_sequence)
            x = params['x']
            y = params['y']
            #if self.half_size_npcs and is_npc:
            #    y -= int(self.image_height /4) - 2
            box = frame[y - shift: y + self.image_height - shift,
                  x: x + self.image_width]
            cv2.imshow('{}_{}'.format(self.identifier, slot_name), box)
            if self.half_size_npcs and is_npc:
                shift += int(self.image_height /2) - int(self.margin)
                show = True
        if show:
            print(kf)
            cv2.waitKey()

    def process_frame(self, frame, time_point, frame_ind):
        if not self.generate_data:
            return
        for rd in self.ranges:
            if rd['begin'] <= time_point <= rd['end']:
                break
        else:
            return

        kf = construct_kf_at_time(self.states, time_point)
        shift = 0
        for slot in self.slots:
            params = self.slot_params[slot]
            if slot > len(kf) - 1:
                is_npc = False
            else:
                is_npc = kf[slot]['is_npc']
            if slot > len(kf) - 1 or kf[slot]['second_hero'] == na_lab:
                sequence, raw_sequence = [], []
            else:
                sequence, raw_sequence = self.lookup_data(kf[slot], time_point)

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
                if self.debug:
                    if raw_sequence:
                        filename = '{}_{}.jpg'.format(' '.join(raw_sequence), self.process_index)
                        cv2.imwrite(os.path.join(self.training_directory, 'debug', pre,
                                             filename), box)
                box = np.transpose(box, axes=(2, 0, 1))
                self.data["{}_img".format(pre)][index, ...] = box[None]
                self.data["{}_round".format(pre)][index] = self.current_round_id
                self.data["{}_time_point".format(pre)][index] = time_point
                # self.train_mean += box / self.hdf5_file['train_img'].shape[0]
                sequence_length = len(sequence)
                self.data['{}_spectator_mode_label'.format(pre)][index] = SPECTATOR_MODES.index(self.spec_mode)
                #if time_point > 262:
                #    print(sequence)
                #    print(raw_sequence)
                #    cv2.imshow('frame', np.transpose(box, (1, 2, 0)))
                #    cv2.waitKey()
                if sequence:
                    self.data["{}_label_sequence_length".format(pre)][index] = sequence_length
                    self.data["{}_label_sequence".format(pre)][index, 0:len(sequence)] = sequence
                    if self.half_size_npcs and is_npc:
                        label = 'half_sized'
                    else:
                        label = 'full_sized'
                else:
                    label = 'empty'
                self.data["{}_exist_label".format(pre)][index] = self.exist_label_set.index(label)
                self.process_index += 1
            if self.half_size_npcs and is_npc:
                shift += int(self.image_height /2) - int(self.margin)
        if time_point % 5 == 0:
            self.states = [x for x in self.states if x['time_point'] > time_point - 8]

    def get_data(self, r):
        self.spec_mode = r['spectator_mode'].lower()
        self.states = get_kf_events(r['id'])

    def add_new_round_info(self, r):
        self.current_round_id = r['id']
        self.hd5_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        if os.path.exists(self.hd5_path) or r['annotation_status'] not in self.usable_annotations:
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

        train_shape = (self.num_train, 3, self.image_height,self.image_width)
        val_shape = (self.num_val, 3, self.image_height, self.image_width)

        self.data = {}
        for pre in ['train', 'val']:
            if pre == 'train':
                shape = train_shape
                count = self.num_train
            else:
                shape = val_shape
                count = self.num_val
            self.data["{}_img".format(pre)] = np.zeros(shape, dtype=np.uint8)
            self.data["{}_round".format(pre)] = np.zeros((count,), dtype=np.int16)
            self.data["{}_time_point".format(pre)] = np.zeros((count,), dtype=np.float)
            self.data["{}_exist_label".format(pre)] = np.zeros((count,), dtype=np.uint8)
            self.data["{}_spectator_mode_label".format(pre)] = np.zeros( (count,), dtype=np.uint8)
            self.data["{}_label_sequence".format(pre)] = np.zeros((count, self.max_sequence_length),
                                          dtype=np.int16)
            self.data["{}_label_sequence".format(pre)][:, :] = len(self.label_set)
            self.data["{}_label_sequence_length".format(pre)] = np.ones((count,), dtype=np.uint8)

        self.process_index = 0