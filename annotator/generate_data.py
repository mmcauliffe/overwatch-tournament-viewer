import requests
import os
import numpy as np
import h5py
import random
import cv2
import shutil
from annotator.utils import get_local_file, HERO_SET, ABILITY_SET, BOX_PARAMETERS, SPECTATOR_MODES, PLAYER_SET, \
    get_player_states, get_local_path, MAP_SET, MAP_MODE_SET, \
    get_kf_events, get_round_states, get_train_rounds, FileVideoStream, look_up_player_state, look_up_round_state, \
    load_set, Empty, calculate_ability_boundaries, calculate_first_hero_boundaries, calculate_hero_boundaries, \
    FileVideoStreamRange, get_event_ranges, COLOR_SET, calculate_first_player_boundaries, calculate_assist_boundaries, \
    get_vod_path, get_train_vods, get_train_rounds_plus, PLAYER_CHARACTER_SET, get_example_rounds

training_data_directory = r'E:\Data\Overwatch\training_data'

cnn_status_train_dir = os.path.join(training_data_directory, 'player_status_cnn')
ocr_status_train_dir = os.path.join(training_data_directory, 'player_status_ocr')
lstm_status_train_dir = os.path.join(training_data_directory, 'player_status_lstm')
cnn_mid_train_dir = os.path.join(training_data_directory, 'mid_cnn')
lstm_mid_train_dir = os.path.join(training_data_directory, 'mid_lstm')
cnn_kf_train_dir = os.path.join(training_data_directory, 'kf_cnn')
lstm_kf_train_dir = os.path.join(training_data_directory, 'kf_lstm')

na_lab = 'n/a'
sides = ['left', 'right']


class DataGenerator(object):
    identifier = ''
    num_variations = 2
    time_step = 0.1
    num_slots = 1
    resize_factor = 1

    def __init__(self):
        self.training_directory = os.path.join(training_data_directory, self.identifier)
        os.makedirs(self.training_directory, exist_ok=True)
        self.rounds_analyzed_path = os.path.join(self.training_directory, 'rounds.txt')
        self.analyzed_rounds = []
        if os.path.exists(self.rounds_analyzed_path):
            with open(self.rounds_analyzed_path, 'r') as f:
                for line in f:
                    self.analyzed_rounds.append(int(line.strip()))
        self.hd5_path = os.path.join(self.training_directory, 'dataset.hdf5')
        self.generate_data = False
        self.hdf5_file = None
        self.states = None
        self.slots = []
        self.slot_params = {}
        self.sets = {}
        self.end_sets = {}
        self.immutable_sets = {}

    def save_set_info(self):
        for k, s in self.sets.items():
            with open(os.path.join(self.training_directory, '{}_set.txt'.format(k)), 'w', encoding='utf8') as f:
                for p in s:
                    f.write('{}\n'.format(p))
        for k, s in self.end_sets.items():
            with open(os.path.join(self.training_directory, '{}_set.txt'.format(k)), 'w', encoding='utf8') as f:
                for p in s:
                    f.write('{}\n'.format(p))
        for k, s in self.immutable_sets.items():
            with open(os.path.join(self.training_directory, '{}_set.txt'.format(k)), 'w', encoding='utf8') as f:
                for p in s:
                    f.write('{}\n'.format(p))

    def figure_slot_params(self, r):
        pass

    def cleanup_round(self):
        if not self.generate_data:
            return
        with open(self.rounds_analyzed_path, 'w') as f:
            for r in self.analyzed_rounds:
                f.write('{}\n'.format(r))
        self.hdf5_file.close()


class CTCDataGenerator(DataGenerator):
    max_sequence_length = 12

    def __init__(self):
        super(CTCDataGenerator, self).__init__()
        self.label_set = []
        self.image_width = 0
        self.image_height = 0
        self.slots = []
        self.thresholds = {'ability': 170,
                           'china contenders': 170,
                           'color': 170,
                           'contenders': 170,
                           'korea contenders': 170,
                           'original': 170,
                           'overwatch league': 170,
                           'world cup': 150}

        self.spectator_modes = [na_lab] + SPECTATOR_MODES
        with open(os.path.join(self.training_directory, 'spectator_mode_set.txt'), 'w', encoding='utf8') as f:
            for p in self.spectator_modes:
                f.write('{}\n'.format(p))

    def save_label_set(self):
        with open(os.path.join(self.training_directory, 'labels_set.txt'), 'w', encoding='utf8') as f:
            for c in self.label_set:
                f.write('{}\n'.format(c))

    def add_new_round_info(self, r):
        if r['id'] in self.analyzed_rounds:
            self.generate_data = False
            return
        self.generate_data = True
        self.figure_slot_params(r)
        self.analyzed_rounds.append(r['id'])
        self.spec_mode = r['spectator_mode'].lower()

        num_frames = 0
        for beg, end in r['sequences']:
            print(beg, end)
            expected_frame_count = int((end - beg) / self.time_step)
            num_frames += (int(expected_frame_count) + 1) * self.num_slots

        num_frames *= self.num_variations
        self.indexes = random.sample(range(num_frames), num_frames)
        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train

        train_shape = (
            self.num_train, int(self.image_width * self.resize_factor), int(self.image_height * self.resize_factor))
        val_shape = (
            self.num_val, int(self.image_width * self.resize_factor), int(self.image_height * self.resize_factor))
        if not os.path.exists(self.hd5_path):
            self.hdf5_file = h5py.File(self.hd5_path, mode='w')
            self.prev_train = 0
            self.prev_val = 0
            for pre in ['train', 'val']:
                if pre == 'train':
                    shape = train_shape
                    count = self.num_train
                else:
                    shape = val_shape
                    count = self.num_val
                self.hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8,
                                              maxshape=(None, shape[1], shape[2]))
                self.hdf5_file.create_dataset("{}_label_sequence".format(pre), (count, self.max_sequence_length),
                                              np.uint32, maxshape=(None, self.max_sequence_length),
                                              fillvalue=len(self.label_set))
                self.hdf5_file.create_dataset("{}_label_sequence_length".format(pre), (count,), np.uint8,
                                              maxshape=(None,), fillvalue=1)
                self.hdf5_file.create_dataset("{}_spectator_mode".format(pre), (count,), np.uint8, maxshape=(None,))
                self.hdf5_file["{}_spectator_mode".format(pre)][0:count] = self.spectator_modes.index(self.spec_mode)
        else:
            self.hdf5_file = h5py.File(self.hd5_path, mode='a')
            self.prev_train = self.hdf5_file['train_img'].shape[0]
            self.prev_val = self.hdf5_file['val_img'].shape[0]
            for pre in ['train', 'val']:
                if pre == 'train':
                    old_count = self.prev_train
                    new_count = self.prev_train + self.num_train
                else:
                    old_count = self.prev_val
                    new_count = self.prev_val + self.num_val
                self.hdf5_file["{}_img".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_label_sequence".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_label_sequence_length".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_spectator_mode".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_spectator_mode".format(pre)][old_count:new_count] = self.spectator_modes.index(
                    self.spec_mode)
        self.process_index = 0

    def process_frame(self, frame, time_point):
        #cv2.imshow('frame', frame)
        for s in self.slots:
            params = self.slot_params[s]
            sequence = self.lookup_data(s, time_point)

            variation_set = [(0, 0)]
            while len(variation_set) < self.num_variations:
                x_offset = random.randint(-3, 3)
                y_offset = random.randint(-3, 3)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))

            x = params['x']
            y = params['y']
            thresh = None
            for i, (x_offset, y_offset) in enumerate(variation_set):
                index = self.indexes[self.process_index]
                if index < self.num_train:
                    pre = 'train'
                    index += self.prev_train
                else:
                    pre = 'val'
                    index -= self.num_train
                    index += self.prev_val
                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
                if thresh is None:
                    thresh, bw = cv2.threshold(gray, self.thresholds[self.spec_mode], 255,
                                   cv2.THRESH_BINARY| cv2.THRESH_OTSU)
                else:
                    bw = cv2.threshold(gray, self.thresholds[self.spec_mode], 255,
                                   cv2.THRESH_BINARY)[1]
                #if i == 0:
                    #cv2.imshow('gray_{}'.format(s), gray)
                    #cv2.imshow('bw_{}'.format(s), bw)

                self.hdf5_file["{}_img".format(pre)][index, ...] = np.swapaxes(bw, 1, 0)[None]
                sequence_length = len(sequence)
                if sequence:
                    self.hdf5_file["{}_label_sequence_length".format(pre)][index] = sequence_length
                    self.hdf5_file["{}_label_sequence".format(pre)][index, 0:len(sequence)] = sequence
                self.process_index += 1
        #cv2.waitKey(0)

class SequenceDataGenerator(DataGenerator):
    frames_per_seq = 100

    def __init__(self):
        super(SequenceDataGenerator, self).__init__()
        self.sets = {}
        self.end_sets = {}
        self.immutable_sets = {}
        self.variation_sets = {}
        self.immutable_set_values = {}
        self.images = {}
        self.data = {}
        self.process_index = 0
        self.current_sequence_index = 0
        self.time_point = 0
        self.slots = []

    def add_new_round_info(self, r):
        if r['id'] in self.analyzed_rounds:
            self.generate_data = False
            return
        self.analyzed_rounds.append(r['id'])
        self.generate_data = True
        self.figure_slot_params(r)
        self.num_sequences = 0
        for beg, end in r['sequences']:
            print(beg, end)
            expected_frame_count = int((end - beg) / self.time_step)
            self.num_sequences += (int(expected_frame_count / self.frames_per_seq) + 1) * self.num_slots
        self.current_sequence_index = 0
        self.num_sequences *= self.num_variations
        self.indexes = random.sample(range(self.num_sequences), self.num_sequences)
        self.num_train = int(self.num_sequences * 0.8)
        self.num_val = self.num_sequences - self.num_train
        train_shape = (self.num_train, self.frames_per_seq, int(self.image_height * self.resize_factor),
                       int(self.image_width * self.resize_factor), 3)
        val_shape = (self.num_val, self.frames_per_seq, int(self.image_height * self.resize_factor),
                     int(self.image_width * self.resize_factor), 3)
        if not os.path.exists(self.hd5_path):
            self.prev_train = 0
            self.prev_val = 0
            self.hdf5_file = h5py.File(self.hd5_path, mode='w')
            for pre in ['train', 'val']:
                if pre == 'train':
                    shape = train_shape
                    num = self.num_train
                else:
                    shape = val_shape
                    num = self.num_val
                self.hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8,
                                              maxshape=(None, shape[1], shape[2], shape[3], shape[4]))
                self.hdf5_file.create_dataset("{}_round".format(pre), (num,), np.uint32, maxshape=(None,))
                self.hdf5_file.create_dataset("{}_time_point".format(pre), (num,), np.float32, maxshape=(None,))
                for k in self.sets.keys():
                    self.hdf5_file.create_dataset("{}_{}_label".format(pre, k), (num, self.frames_per_seq), np.uint8,
                                                  maxshape=(None, self.frames_per_seq))
                for k, s in self.end_sets.items():
                    self.hdf5_file.create_dataset("{}_{}_label".format(pre, k), (num,), np.uint8, maxshape=(None,))
                for k, s in self.immutable_sets.items():
                    self.hdf5_file.create_dataset("{}_{}_label".format(pre, k), (num,), np.uint8, maxshape=(None,))
                    self.hdf5_file["{}_{}_label".format(pre, k)][0:num] = s.index(self.immutable_set_values[k])

        else:
            self.hdf5_file = h5py.File(self.hd5_path, mode='a')
            self.prev_train = self.hdf5_file['train_img'].shape[0]
            self.prev_val = self.hdf5_file['val_img'].shape[0]
            for pre in ['train', 'val']:
                if pre == 'train':
                    old_count = self.prev_train
                    new_count = self.prev_train + self.num_train
                else:
                    old_count = self.prev_val
                    new_count = self.prev_val + self.num_val
                self.hdf5_file["{}_img".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_round".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_time_point".format(pre)].resize(new_count, axis=0)
                for k in self.sets.keys():
                    self.hdf5_file["{}_{}_label".format(pre, k)].resize(new_count, axis=0)
                for k, s in self.end_sets.items():
                    self.hdf5_file["{}_{}_label".format(pre, k)].resize(new_count, axis=0)
                for k, s in self.immutable_sets.items():
                    self.hdf5_file["{}_{}_label".format(pre, k)].resize(new_count, axis=0)
                    self.hdf5_file["{}_{}_label".format(pre, k)][old_count:new_count] = s.index(
                        self.immutable_set_values[k])
        self.reset_cached_data()

    def process_frame(self, frame, time_point):
        if not self.generate_data:
            return
        if self.current_sequence_index >= len(self.indexes):
            self.current_sequence_index += 1
            return
        self.time_point = time_point
        for slot in self.slots:
            d = self.lookup_data(slot, time_point)
            params = self.slot_params[slot]
            x = params['x']
            y = params['y']
            for i, (x_offset, y_offset) in enumerate(self.variation_sets[slot]):
                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                if self.resize_factor != 1:
                    box = cv2.resize(box, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
                self.images[slot][i, self.process_index, ...] = box[None]

            for k in self.sets.keys():
                self.data[slot][k][self.process_index] = self.sets[k].index(d[k])
            for k in self.end_sets.keys():
                self.data[slot][k] = self.end_sets[k].index(d[k])

        self.process_index += 1
        if self.process_index == self.frames_per_seq:
            self.save_current_sequence()

    def reset_cached_data(self):
        self.variation_sets = {}
        self.images = {}
        self.data = {}
        self.process_index = 0

        for slot in self.slots:
            self.variation_sets[slot] = [(0, 0)]
            while len(self.variation_sets[slot]) < self.num_variations:
                x_offset = random.randint(-5, 5)
                y_offset = random.randint(-5, 5)
                if (x_offset, y_offset) in self.variation_sets[slot]:
                    continue
                self.variation_sets[slot].append((x_offset, y_offset))
            self.images[slot] = np.zeros(
                (self.num_variations, self.frames_per_seq, int(self.image_height * self.resize_factor),
                 int(self.image_width * self.resize_factor), 3),
                dtype=np.uint8)
            self.data[slot] = {}
            for k in self.sets.keys():
                self.data[slot][k] = np.zeros((self.frames_per_seq,), dtype=np.uint8)
            for k in self.end_sets.keys():
                self.data[slot][k] = 0

    def save_current_sequence(self):
        if not self.generate_data:
            return
        if self.process_index == 0:
            return
        for slot in self.slots:
            for i in range(self.num_variations):
                index = self.indexes[self.current_sequence_index]
                if index < self.num_train:
                    pre = 'train'
                    index += self.prev_train
                else:
                    pre = 'val'
                    index -= self.num_train
                    index += self.prev_val
                self.hdf5_file["{}_img".format(pre)][index, ...] = self.images[slot][i, ...]
                self.hdf5_file["{}_round".format(pre)][index] = r['id']
                self.hdf5_file["{}_time_point".format(pre)][index] = self.time_point

                for k in self.sets.keys():
                    self.hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = self.data[slot][k][None]
                for k in self.end_sets.keys():
                    self.hdf5_file['{}_{}_label'.format(pre, k)][index] = self.data[slot][k]
                self.current_sequence_index += 1
        self.reset_cached_data()

    def cleanup_round(self):
        if not self.generate_data:
            return
        with open(self.rounds_analyzed_path, 'w') as f:
            for r in self.analyzed_rounds:
                f.write('{}\n'.format(r))
        self.save_current_sequence()
        self.hdf5_file.close()


class KillFeedCTCGenerator(CTCDataGenerator):
    identifier = 'kill_feed_ctc'
    num_slots = 6

    def __init__(self):
        super(KillFeedCTCGenerator, self).__init__()
        self.label_set = COLOR_SET + HERO_SET + [x + '_assist' for x in HERO_SET] + ABILITY_SET
        self.save_label_set()
        self.image_width = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['WIDTH']
        self.image_height = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['HEIGHT']
        self.slots = range(6)

    def figure_slot_params(self, r):
        self.slot_params = {}
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['KILL_FEED_SLOT']
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

        sequence.append(self.label_set.index(d['second_hero']))
        sequence.append(self.label_set.index(d['second_color']))
        return sequence

    def process_frame(self, frame, time_point):
        if not self.generate_data:
            return
        for rd in self.ranges:
            if rd['begin'] <= time_point <= rd['end']:
                break
        else:
            return

        kf, e = construct_kf_at_time(self.states, time_point)
        for s in self.slots:
            params = self.slot_params[s]
            if s > len(kf) - 1 or kf[s]['second_hero'] == na_lab:
                sequence = []
            else:
                sequence = self.lookup_data(kf[s], time_point)

            variation_set = [(0, 0)]
            while len(variation_set) < self.num_variations:
                x_offset = random.randint(-5, 5)
                y_offset = random.randint(-5, 5)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))

            x = params['x']
            y = params['y']

            for i, (x_offset, y_offset) in enumerate(variation_set):
                if self.process_index > len(self.indexes) - 1:
                    continue
                index = self.indexes[self.process_index]
                if index < self.num_train:
                    pre = 'train'
                    index += self.prev_train
                else:
                    pre = 'val'
                    index -= self.num_train
                    index += self.prev_val
                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                self.hdf5_file["{}_img".format(pre)][index, ...] = np.swapaxes(box, 1, 0)[None]
                sequence_length = len(sequence)
                if sequence:
                    self.hdf5_file["{}_label_sequence_length".format(pre)][index] = sequence_length
                    self.hdf5_file["{}_label_sequence".format(pre)][index, 0:len(sequence)] = sequence
                self.process_index += 1

    def add_new_round_info(self, r):
        if r['id'] in self.analyzed_rounds:
            self.generate_data = False
            return
        self.generate_data = True
        self.states = get_kf_events(r['id'])
        self.ranges = get_event_ranges(self.states, r['end'] - r['begin'])
        self.figure_slot_params(r)
        self.analyzed_rounds.append(r['id'])
        self.spec_mode = r['spectator_mode'].lower()

        num_frames = 0
        for rd in self.ranges:
            expected_duration = rd['end'] - rd['begin']
            expected_frame_count = expected_duration / self.time_step
            num_frames += (int(expected_frame_count) + 1) * self.num_slots

        num_frames *= self.num_variations
        self.indexes = random.sample(range(num_frames), num_frames)
        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train

        train_shape = (self.num_train, self.image_width, self.image_height, 3)
        val_shape = (self.num_val, self.image_width, self.image_height, 3)
        if not os.path.exists(self.hd5_path):
            self.hdf5_file = h5py.File(self.hd5_path, mode='w')
            self.prev_train = 0
            self.prev_val = 0
            for pre in ['train', 'val']:
                if pre == 'train':
                    shape = train_shape
                    count = self.num_train
                else:
                    shape = val_shape
                    count = self.num_val
                self.hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8,
                                              maxshape=(None, shape[1], shape[2], shape[3]))
                self.hdf5_file.create_dataset("{}_label_sequence".format(pre), (count, self.max_sequence_length),
                                              np.uint32, maxshape=(None, self.max_sequence_length),
                                              fillvalue=len(self.label_set))
                self.hdf5_file.create_dataset("{}_label_sequence_length".format(pre), (count,), np.uint8,
                                              maxshape=(None,), fillvalue=1)
                self.hdf5_file.create_dataset("{}_spectator_mode".format(pre), (count,), np.uint8, maxshape=(None,))
                self.hdf5_file["{}_spectator_mode".format(pre)][0:count] = self.spectator_modes.index(self.spec_mode)
        else:
            self.hdf5_file = h5py.File(self.hd5_path, mode='a')
            self.prev_train = self.hdf5_file['train_img'].shape[0]
            self.prev_val = self.hdf5_file['val_img'].shape[0]
            for pre in ['train', 'val']:
                if pre == 'train':
                    old_count = self.prev_train
                    new_count = self.prev_train + self.num_train
                else:
                    old_count = self.prev_val
                    new_count = self.prev_val + self.num_val
                self.hdf5_file["{}_img".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_label_sequence".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_label_sequence_length".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_spectator_mode".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_spectator_mode".format(pre)][old_count:new_count] = self.spectator_modes.index(
                    self.spec_mode)
        self.process_index = 0


class PlayerOCRGenerator(CTCDataGenerator):
    identifier = 'player_ocr'
    num_slots = 12
    time_step = 2
    num_variations = 5

    def __init__(self):
        super(PlayerOCRGenerator, self).__init__()
        self.label_set = PLAYER_CHARACTER_SET
        self.save_label_set()
        self.sets = {'alive': [na_lab, 'alive', 'dead'],
                     'color': [na_lab] + COLOR_SET}
        self.save_set_info()

        params = BOX_PARAMETERS['O']['LEFT_NAME']
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        for side in sides:
            for i in range(6):
                self.slots.append((side, i))

    def lookup_data(self, slot, time_point):
        d = look_up_player_state(slot[0], slot[1], time_point, self.states)
        return [self.label_set.index(x) for x in self.names[slot]], d['alive'], self.colors[slot]

    def process_frame(self, frame, time_point):
        #cv2.imshow('frame', frame)
        for s in self.slots:
            params = self.slot_params[s]
            sequence, alive, color = self.lookup_data(s, time_point)

            variation_set = [(0, 0)]
            while len(variation_set) < self.num_variations:
                x_offset = random.randint(-3, 3)
                y_offset = random.randint(-3, 3)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))

            x = params['x']
            y = params['y']

            for i, (x_offset, y_offset) in enumerate(variation_set):
                index = self.indexes[self.process_index]
                if index < self.num_train:
                    pre = 'train'
                    index += self.prev_train
                else:
                    pre = 'val'
                    index -= self.num_train
                    index += self.prev_val
                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                #if i == 0:
                    #cv2.imshow('gray_{}'.format(s), gray)
                    #cv2.imshow('bw_{}'.format(s), bw)

                self.hdf5_file["{}_img".format(pre)][index, ...] = np.swapaxes(box, 1, 0)[None]
                sequence_length = len(sequence)
                if sequence:
                    self.hdf5_file["{}_label_sequence_length".format(pre)][index] = sequence_length
                    self.hdf5_file["{}_label_sequence".format(pre)][index, 0:len(sequence)] = sequence
                    self.hdf5_file["{}_color_label".format(pre)][index] = self.sets['color'].index(color)
                    self.hdf5_file["{}_alive_label".format(pre)][index] = self.sets['alive'].index(alive)
                self.process_index += 1
        #cv2.waitKey(0)

    def add_new_round_info(self, r):
        if r['id'] in self.analyzed_rounds:
            self.generate_data = False
            return
        self.generate_data = True
        self.figure_slot_params(r)
        self.analyzed_rounds.append(r['id'])
        self.spec_mode = r['spectator_mode'].lower()

        num_frames = 0
        for beg, end in r['sequences']:
            print(beg, end)
            expected_frame_count = int((end - beg) / self.time_step)
            num_frames += (int(expected_frame_count) + 1) * self.num_slots

        num_frames *= self.num_variations
        self.indexes = random.sample(range(num_frames), num_frames)
        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train
        self.left_color = r['game']['left_team']['color'].lower()
        self.right_color = r['game']['right_team']['color'].lower()
        train_shape = (
            self.num_train, int(self.image_width * self.resize_factor), int(self.image_height * self.resize_factor), 3)
        val_shape = (
            self.num_val, int(self.image_width * self.resize_factor), int(self.image_height * self.resize_factor), 3)
        if not os.path.exists(self.hd5_path):
            self.hdf5_file = h5py.File(self.hd5_path, mode='w')
            self.prev_train = 0
            self.prev_val = 0
            for pre in ['train', 'val']:
                if pre == 'train':
                    shape = train_shape
                    count = self.num_train
                else:
                    shape = val_shape
                    count = self.num_val
                self.hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8,
                                              maxshape=(None, shape[1], shape[2], shape[3]))
                self.hdf5_file.create_dataset("{}_label_sequence".format(pre), (count, self.max_sequence_length),
                                              np.uint32, maxshape=(None, self.max_sequence_length),
                                              fillvalue=len(self.label_set))
                self.hdf5_file.create_dataset("{}_label_sequence_length".format(pre), (count,), np.uint8,
                                              maxshape=(None,), fillvalue=1)
                self.hdf5_file.create_dataset("{}_spectator_mode_label".format(pre), (count,), np.uint8, maxshape=(None,))
                self.hdf5_file.create_dataset("{}_color_label".format(pre), (count,), np.uint8, maxshape=(None,))
                self.hdf5_file.create_dataset("{}_alive_label".format(pre), (count,), np.uint8, maxshape=(None,))
                self.hdf5_file["{}_spectator_mode_label".format(pre)][0:count] = self.spectator_modes.index(self.spec_mode)
        else:
            self.hdf5_file = h5py.File(self.hd5_path, mode='a')
            self.prev_train = self.hdf5_file['train_img'].shape[0]
            self.prev_val = self.hdf5_file['val_img'].shape[0]
            for pre in ['train', 'val']:
                if pre == 'train':
                    old_count = self.prev_train
                    new_count = self.prev_train + self.num_train
                else:
                    old_count = self.prev_val
                    new_count = self.prev_val + self.num_val
                self.hdf5_file["{}_img".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_label_sequence".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_label_sequence_length".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_color_label".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_alive_label".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_spectator_mode_label".format(pre)].resize(new_count, axis=0)
                self.hdf5_file["{}_spectator_mode_label".format(pre)][old_count:new_count] = self.spectator_modes.index(
                    self.spec_mode)
        self.process_index = 0
        self.states = get_player_states(r['id'])

        self.names = {}
        self.colors = {}
        for slot in self.slots:
            self.names[slot] = self.states[slot[0]][str(slot[1])]['player']
            if slot[0] == 'left':
                self.colors[slot] = self.left_color
            else:
                self.colors[slot] = self.right_color

    def figure_slot_params(self, r):
        left_params = BOX_PARAMETERS[r['stream_vod']['film_format']]['LEFT_NAME']
        right_params = BOX_PARAMETERS[r['stream_vod']['film_format']]['RIGHT_NAME']
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


class MidStatusGenerator(SequenceDataGenerator):
    identifier = 'mid'
    resize_factor = 0.5

    def __init__(self):
        super(MidStatusGenerator, self).__init__()
        self.immutable_sets = {'attacking_color': [na_lab] + COLOR_SET,
                               'map': [na_lab] + MAP_SET,
                               'map_mode': [na_lab] + MAP_MODE_SET,
                               'round_number': range(1, 10),
                               'spectator_mode': [na_lab] + SPECTATOR_MODES}
        self.end_sets = {}
        self.sets = {
            'overtime': [na_lab] + ['not_overtime', 'overtime'],
            'point_status': [na_lab] + sorted(['Assault_A', 'Assault_B',
                                               'Escort_1', 'Escort_2', 'Escort_3'] +
                                              ['Control_' + x for x in self.immutable_sets['attacking_color']]),
        }
        params = BOX_PARAMETERS['O']['MID']
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        self.save_set_info()
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
                    d[k] = 'Control_' + self.immutable_set_values['left_color']
                elif d[k] == 'Control_right':
                    d[k] = 'Control_' + self.immutable_set_values['right_color']

        # d['attacking_color'] = self.attacking_color
        # d['map'] = self.map
        # d['spectator_mode'] = self.spec_mode
        # d['map_mode'] = self.map_mode
        # d['round_number'] = self.round_number
        return d

    def add_new_round_info(self, r):
        self.immutable_set_values = {'left_color': r['game']['left_team']['color'].lower(),
                                     'attacking_color': r['attacking_color'].lower(),
                                     'right_color': r['game']['right_team']['color'].lower(),
                                     'map': r['game']['map']['name'].lower(),
                                     'map_mode': r['game']['map']['mode'].lower(),
                                     'round_number': r['round_number'],
                                     'spectator_mode': r['spectator_mode'].lower()}
        super(MidStatusGenerator, self).add_new_round_info(r)
        if not self.generate_data:
            return

        self.states = get_round_states(r['id'])


class PlayerStatusGenerator(SequenceDataGenerator):
    identifier = 'player_status'
    num_slots = 12

    def __init__(self):
        super(PlayerStatusGenerator, self).__init__()
        self.sets = {'hero': [na_lab] + HERO_SET,
                     'ult': [na_lab, 'no_ult', 'has_ult'],
                     'alive': [na_lab, 'alive', 'dead'],
                     'side': [na_lab] + sides, }
        self.end_sets = {
            # 'player': [na_lab] + PLAYER_SET,
            'color': [na_lab] + COLOR_SET, }
        self.immutable_sets = {'spectator_mode': [na_lab] + SPECTATOR_MODES, }
        self.save_set_info()
        params = BOX_PARAMETERS['O']['LEFT']
        self.image_width = params['WIDTH']
        self.image_height = params['HEIGHT']
        for s in sides:
            for i in range(6):
                self.slots.append((s, i))
        self.slot_params = {}

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
        self.immutable_set_values = {'spectator_mode': r['spectator_mode'].lower()}
        super(PlayerStatusGenerator, self).add_new_round_info(r)
        if not self.generate_data:
            return

        self.states = get_player_states(r['id'])
        self.left_color = r['game']['left_team']['color'].lower()
        self.right_color = r['game']['right_team']['color'].lower()

    def lookup_data(self, slot, time_point):
        d = look_up_player_state(slot[0], slot[1], time_point, self.states)
        if slot[0] == 'left':
            d['color'] = self.left_color
        else:
            d['color'] = self.right_color
        d['side'] = slot[0]
        return d


def generate_data(rounds):
    import time as timepackage
    generators = [PlayerStatusGenerator(), PlayerOCRGenerator(), KillFeedCTCGenerator(), MidStatusGenerator()]
    for round_index, r in enumerate(rounds):
        print("Processing round {} of {}".format(round_index, len(rounds)))
        print(r)
        print(r['spectator_mode'])
        begin_time = timepackage.time()
        process_round = False
        for g in generators:
            g.add_new_round_info(r)
            if g.generate_data:
                process_round = True
        if not process_round:
            continue
        time_step = min(x.time_step for x in generators if x.generate_data)
        for beg, end in r['sequences']:
            print(beg, end)
            fvs = FileVideoStream(get_vod_path(r['stream_vod']), beg + r['begin'], end + r['begin'], time_step,
                                  real_begin=r['begin']).start()
            timepackage.sleep(1.0)
            frame_ind = 0
            num_frames = int((end - beg) / time_step)
            while True:
                try:
                    frame, time_point = fvs.read()
                except Empty:
                    break
                if frame_ind % 100 == 0:
                    print('Frame: {}/{}'.format(frame_ind, num_frames))
                for g in generators:
                    if time_step == g.time_step:
                        g.process_frame(frame, time_point)
                    elif frame_ind % (g.time_step / time_step) == 0:
                        g.process_frame(frame, time_point)
                frame_ind += 1
            for g in generators:
                if isinstance(g, SequenceDataGenerator):
                    g.save_current_sequence()

        for g in generators:
            g.cleanup_round()
        print('Finished in {} seconds!'.format(timepackage.time() - begin_time))


def construct_kf_at_time(events, time):
    window = 7.3
    possible_kf = []
    event_at_time = False
    for e in events:
        if e['time_point'] > time + 0.25:
            break
        elif e['time_point'] > time:
            possible_kf.insert(0, {'time_point': e['time_point'],
                                   'first_hero': 'n/a', 'first_color': 'n/a', 'ability': 'n/a', 'headshot': 'n/a',
                                   'second_hero': 'n/a',
                                   'second_color': 'n/a', })
        if time - window <= e['time_point'] <= time:
            if abs(time - e['time_point']) < 0.05:
                event_at_time = True
            for k, v in e.items():
                if isinstance(v, str):
                    e[k] = v.lower()
                # if 'color' in k:
                #    if e[k] != 'white':
                #        e[k] = 'nonwhite'
            possible_kf.append(e)
    possible_kf = sorted(possible_kf, key=lambda x: -1 * x['time_point'])
    return possible_kf[:6], event_at_time


def generate_data_for_replay_cnn(rounds):
    train_dir = os.path.join(training_data_directory, 'replay_cnn')
    import time as timepackage
    debug = False
    time_step = 0.1
    hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    os.makedirs(train_dir, exist_ok=True)
    if os.path.exists(hd5_path):
        print('skipping replay cnn data')
        return
    print('beginning replay cnn data')

    # calc params
    num_frames = 0
    states = {}
    for r in rounds:
        states[r['id']] = get_round_states(r['id'])
        if len(states[r['id']]['replays']) == 1:
            continue
        for s in states[r['id']]['replays']:
            if s['status'] == 'replay':
                expected_frame_count = int((s['end'] - s['begin']) / time_step)
                print(expected_frame_count)
                num_frames += (int(expected_frame_count) + 1) * 2

    na_lab = 'n/a'
    replay_set = [na_lab, 'not_replay', 'replay']
    spectator_modes = [na_lab] + SPECTATOR_MODES

    print(num_frames)
    indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train

    params = BOX_PARAMETERS['O']['REPLAY']

    train_shape = (num_train, params['HEIGHT'], params['WIDTH'], 3)
    val_shape = (num_val, params['HEIGHT'], params['WIDTH'], 3)

    hdf5_file = h5py.File(hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        hdf5_file.create_dataset("{}_spectator_mode".format(pre), (count,), np.uint8)
        hdf5_file.create_dataset("{}_label".format(pre), (count,), np.uint8)

    frame_ind = 0
    for r in rounds:
        if len(states[r['id']]['replays']) == 1:
            continue

        spec_mode = r['spectator_mode'].lower()
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['REPLAY']
        for s in states[r['id']]['replays']:
            if s['status'] == 'replay':
                duration = s['end'] - s['begin']
                beg = s['begin'] - duration
                end = s['end'] + duration
                print(beg, end)
                fvs = FileVideoStream(get_vod_path(r['stream_vod']), beg + r['begin'], end + r['begin'], time_step,
                                      real_begin=r['begin']).start()
                timepackage.sleep(1.0)
                while True:
                    try:
                        frame, time_point = fvs.read()
                    except Empty:
                        break
                    time_point = round(time_point, 1)
                    if frame_ind >= len(indexes):
                        print('ignoring')
                        frame_ind += 1
                        continue

                    index = indexes[frame_ind]
                    if index < num_train:
                        pre = 'train'
                    else:
                        pre = 'val'
                        index -= num_train
                    if frame_ind != 0 and (frame_ind) % 100 == 0 and frame_ind < num_train:
                        print('Train data: {}/{}'.format(frame_ind, num_train))
                    elif frame_ind != 0 and frame_ind % 100 == 0:
                        print('Validation data: {}/{}'.format(frame_ind - num_train, num_val))
                    lab = 'not_replay'
                    if s['begin'] <= time_point <= s['end']:
                        lab = 'replay'
                    x = params['X']
                    y = params['Y']
                    box = frame[y: y + params['HEIGHT'],
                          x: x + params['WIDTH']]
                    if debug and lab == 'replay':
                        cv2.imshow('frame', frame)
                        cv2.imshow('box', box)
                        cv2.waitKey(0)
                    hdf5_file["{}_img".format(pre)][index, ...] = box[None]
                    hdf5_file["{}_round".format(pre)][index] = r['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    hdf5_file["{}_spectator_mode".format(pre)][index] = spectator_modes.index(spec_mode)
                    hdf5_file["{}_label".format(pre)][index] = replay_set.index(lab)

    hdf5_file.close()
    with open(os.path.join(train_dir, 'replay_set.txt'), 'w', encoding='utf8') as f:
        for p in replay_set:
            f.write('{}\n'.format(p))
    with open(os.path.join(train_dir, 'spectator_mode_set.txt'), 'w', encoding='utf8') as f:
        for p in spectator_modes:
            f.write('{}\n'.format(p))


def generate_data_for_pause_cnn(rounds):
    train_dir = os.path.join(training_data_directory, 'pause_cnn')
    import time as timepackage
    debug = False
    time_step = 0.1
    hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    os.makedirs(train_dir, exist_ok=True)
    if os.path.exists(hd5_path):
        print('skipping pause cnn data')
        return
    print('beginning pause cnn data')

    # calc params
    num_frames = 0
    states = {}
    for r in rounds:
        states[r['id']] = get_round_states(r['id'])
        print(states[r['id']])
        if len(states[r['id']]['pauses']) == 1:
            continue
        for s in states[r['id']]['pauses']:
            if s['status'] == 'paused':
                expected_frame_count = int((s['end'] - s['begin']) / time_step)
                print(expected_frame_count)
                num_frames += (int(expected_frame_count) + 1) * 2
    na_lab = 'n/a'
    pause_set = [na_lab, 'not_paused', 'paused']
    spectator_modes = [na_lab] + SPECTATOR_MODES

    print(num_frames)
    indexes = random.sample(range(num_frames), num_frames)
    num_train = int(num_frames * 0.8)
    num_val = num_frames - num_train

    params = BOX_PARAMETERS['O']['PAUSE']

    train_shape = (num_train, params['HEIGHT'], params['WIDTH'], 3)
    val_shape = (num_val, params['HEIGHT'], params['WIDTH'], 3)

    hdf5_file = h5py.File(hd5_path, mode='w')
    for pre in ['train', 'val']:
        count = num_train
        shape = train_shape
        if pre == 'val':
            count = num_val
            shape = val_shape
        hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
        hdf5_file.create_dataset("{}_round".format(pre), (count,), np.uint32)
        hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32)
        hdf5_file.create_dataset("{}_spectator_mode".format(pre), (count,), np.uint8)
        hdf5_file.create_dataset("{}_label".format(pre), (count,), np.uint8)

    frame_ind = 0
    for r in rounds:
        if len(states[r['id']]['pauses']) == 1:
            continue
        spec_mode = r['spectator_mode'].lower()
        params = BOX_PARAMETERS[r['stream_vod']['film_format']]['PAUSE']
        for s in states[r['id']]['pauses']:
            if s['status'] == 'paused':
                duration = s['end'] - s['begin']
                beg = s['begin'] - duration
                end = s['end'] + duration
                print(beg, end)
                fvs = FileVideoStream(get_vod_path(r['stream_vod']), beg + r['begin'], end + r['begin'], time_step,
                                      real_begin=r['begin']).start()
                timepackage.sleep(1.0)
                while True:
                    try:
                        frame, time_point = fvs.read()
                    except Empty:
                        break
                    time_point = round(time_point, 1)
                    if frame_ind >= len(indexes):
                        print('ignoring')
                        frame_ind += 1
                        continue

                    index = indexes[frame_ind]
                    if index < num_train:
                        pre = 'train'
                    else:
                        pre = 'val'
                        index -= num_train
                    if frame_ind != 0 and (frame_ind) % 100 == 0 and frame_ind < num_train:
                        print('Train data: {}/{}'.format(frame_ind, num_train))
                    elif frame_ind != 0 and frame_ind % 100 == 0:
                        print('Validation data: {}/{}'.format(frame_ind - num_train, num_val))
                    lab = 'not_paused'
                    if s['begin'] <= time_point <= s['end']:
                        lab = 'paused'
                    x = params['X']
                    y = params['Y']
                    box = frame[y: y + params['HEIGHT'],
                          x: x + params['WIDTH']]
                    if debug and lab == 'paused':
                        cv2.imshow('frame', frame)
                        cv2.imshow('box', box)
                        cv2.waitKey(0)
                    hdf5_file["{}_img".format(pre)][index, ...] = box[None]
                    hdf5_file["{}_round".format(pre)][index] = r['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    hdf5_file["{}_spectator_mode".format(pre)][index] = spectator_modes.index(spec_mode)
                    hdf5_file["{}_label".format(pre)][index] = pause_set.index(lab)

    hdf5_file.close()
    with open(os.path.join(train_dir, 'pause_set.txt'), 'w', encoding='utf8') as f:
        for p in pause_set:
            f.write('{}\n'.format(p))
    with open(os.path.join(train_dir, 'spectator_mode_set.txt'), 'w', encoding='utf8') as f:
        for p in spectator_modes:
            f.write('{}\n'.format(p))


def generate_data_for_game_cnn(vods):
    train_dir = os.path.join(training_data_directory, 'game_cnn')
    import time as timepackage
    debug = False
    time_step = 1
    os.makedirs(train_dir, exist_ok=True)
    generated_vods_path = os.path.join(train_dir, 'vods.txt')
    analyzed_vods = []
    labels = ['not_in_game', 'game']
    if os.path.exists(generated_vods_path):
        with open(generated_vods_path, 'r') as f:
            for line in f:
                analyzed_vods.append(int(line.strip()))
    hd5_path = os.path.join(train_dir, 'dataset.hdf5')
    with open(os.path.join(train_dir, 'labels.txt'), 'w', encoding='utf8') as f:
        for p in labels:
            f.write('{}\n'.format(p))

    print('beginning game cnn data')
    error_set = []
    frames_per_seq = 100
    num_variations = 5
    resize_factor = 0.5
    seqs = {}
    print('analyzed', analyzed_vods)
    for v_i, v in enumerate(vods):
        print(v_i, len(vods))
        print(v)
        if v['id'] in analyzed_vods:
            continue
        analyzed_vods.append(v['id'])
        seqs[v['id']] = []
        cap = cv2.VideoCapture(get_vod_path(v))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur = num_frames / fps
        cap.release()
        num_sequences = 0
        num_frames = 0
        for i, s in enumerate(v['sequences']):
            beg, end = s
            seq_dur = end - beg
            non_seq_dur = seq_dur / 2
            prev_beg = beg - non_seq_dur
            if i == 0 and prev_beg < 0:
                prev_beg = 0
            elif i > 0 and prev_beg < v['sequences'][i - 1][1]:
                prev_beg = v['sequences'][i - 1][1]
            prev_dur = beg - prev_beg
            foll_end = end + non_seq_dur
            if i == len(v['sequences']) - 1 and foll_end > dur:
                foll_end = dur
            elif i < len(v['sequences']) - 1 and foll_end > v['sequences'][i + 1][0]:
                foll_end = v['sequences'][i + 1][0]
            foll_dur = foll_end - end
            expected_frame_count = int((seq_dur + prev_dur + foll_dur) / time_step)

            num_frames += (int(expected_frame_count) + 1)
            num_sequences += (int(expected_frame_count / frames_per_seq) + 1)
            seqs[v['id']].append([prev_beg, foll_end])
        num_sequences *= num_variations
        print(num_sequences)
        indexes = random.sample(range(num_sequences), num_sequences)
        num_train = int(num_sequences * 0.8)
        num_val = num_sequences - num_train

        params = BOX_PARAMETERS['O']['MID']
        train_shape = (
            num_train, frames_per_seq, int(params['HEIGHT'] * resize_factor), int(params['WIDTH'] * resize_factor), 3)
        val_shape = (
            num_val, frames_per_seq, int(params['HEIGHT'] * resize_factor), int(params['WIDTH'] * resize_factor), 3)

        if not os.path.exists(hd5_path):
            prev_train = 0
            prev_val = 0
            hdf5_file = h5py.File(hd5_path, mode='w')
            for pre in ['train', 'val']:
                count = num_train
                shape = train_shape
                if pre == 'val':
                    count = num_val
                    shape = val_shape
                hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8,
                                         maxshape=(None, shape[1], shape[2], shape[3], shape[4]))
                hdf5_file.create_dataset("{}_vod".format(pre), (count,), np.uint32, maxshape=(None,))
                hdf5_file.create_dataset("{}_time_point".format(pre), (count,), np.float32, maxshape=(None,))
                hdf5_file.create_dataset("{}_label".format(pre), (count, frames_per_seq), np.uint8,
                                         maxshape=(None, frames_per_seq))
                # hdf5_file.create_dataset("{}_prev_label".format(pre), (count,), np.uint8)
        else:
            hdf5_file = h5py.File(hd5_path, mode='a')
            prev_train = hdf5_file['train_img'].shape[0]
            prev_val = hdf5_file['val_img'].shape[0]
            for pre in ['train', 'val']:
                if pre == 'train':
                    new_count = prev_train + num_train
                else:
                    new_count = prev_val + num_val
                hdf5_file["{}_img".format(pre)].resize(new_count, axis=0)
                hdf5_file["{}_vod".format(pre)].resize(new_count, axis=0)
                hdf5_file["{}_time_point".format(pre)].resize(new_count, axis=0)
                hdf5_file["{}_label".format(pre)].resize(new_count, axis=0)

        print(num_frames, num_train, num_val)
        sequence_ind = 0
        for seq in seqs[v['id']]:
            beg, end = seq
            print(beg, end)
            fvs = FileVideoStream(get_vod_path(v), beg, end, time_step, real_begin=0).start()
            timepackage.sleep(1.0)
            # prev_label = labels[0]
            begin_time = timepackage.time()
            data = np.zeros((frames_per_seq,), dtype=np.uint8)

            variation_set = [(0, 0)]

            while len(variation_set) < num_variations:
                x_offset = random.randint(-5, 5)
                y_offset = random.randint(-5, 5)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))

            images = np.zeros((num_variations, frames_per_seq, int(params['HEIGHT'] * resize_factor),
                               int(params['WIDTH'] * resize_factor), 3),
                              dtype=np.uint8)

            j = 0
            while True:
                try:
                    frame, time_point = fvs.read()
                except Empty:
                    break
                if sequence_ind >= len(indexes):
                    print('ignoring')
                    sequence_ind += 1
                    continue
                time_point = round(time_point, 1)
                lab = labels[0]
                for s in v['sequences']:
                    if s[0] - 2 <= time_point <= s[1] + 2:
                        lab = labels[1]

                data[j] = labels.index(lab)
                x = params['X']
                y = params['Y']

                for i, (x_offset, y_offset) in enumerate(variation_set):
                    box = frame[y + y_offset: y + params['HEIGHT'] + y_offset,
                          x + x_offset: x + params['WIDTH'] + x_offset]
                    box = cv2.resize(box, (0, 0), fx=resize_factor, fy=resize_factor)
                    images[i, j, ...] = box[None]

                j += 1
                if j == frames_per_seq:
                    for i in range(num_variations):
                        index = indexes[sequence_ind]
                        if index < num_train:
                            pre = 'train'
                            index += prev_train
                        else:
                            pre = 'val'
                            index -= num_train
                            index += prev_val
                        print(sequence_ind, num_sequences)
                        if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                            print('Train data: {}/{}'.format(sequence_ind, num_train))
                        elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                            print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                        hdf5_file["{}_img".format(pre)][index, ...] = images[i, ...]

                        hdf5_file["{}_vod".format(pre)][index] = v['id']
                        hdf5_file["{}_time_point".format(pre)][index] = time_point
                        hdf5_file['{}_label'.format(pre)][index, ...] = data[None]
                        # hdf5_file['{}_prev_label'.format(pre)][index] = labels.index(prev_label)
                        # prev_label = labels[data[-1]]
                        sequence_ind += 1

                    variation_set = [(0, 0)]

                    while len(variation_set) < num_variations:
                        x_offset = random.randint(-5, 5)
                        y_offset = random.randint(-5, 5)
                        if (x_offset, y_offset) in variation_set:
                            continue
                        variation_set.append((x_offset, y_offset))

                    images = np.zeros((num_variations, frames_per_seq, int(params['HEIGHT'] * resize_factor),
                                       int(params['WIDTH'] * resize_factor), 3),
                                      dtype=np.uint8)

                    j = 0
            if j > 0:
                for i in range(num_variations):
                    index = indexes[sequence_ind]
                    if index < num_train:
                        pre = 'train'
                        index += prev_train
                    else:
                        pre = 'val'
                        index -= num_train
                        index += prev_val
                    print(sequence_ind, num_sequences)
                    if sequence_ind != 0 and (sequence_ind) % 100 == 0 and sequence_ind < num_train:
                        print('Train data: {}/{}'.format(sequence_ind, num_train))
                    elif sequence_ind != 0 and sequence_ind % 1000 == 0:
                        print('Validation data: {}/{}'.format(sequence_ind - num_train, num_val))
                    hdf5_file["{}_img".format(pre)][index, ...] = images[i, ...]

                    hdf5_file["{}_vod".format(pre)][index] = v['id']
                    hdf5_file["{}_time_point".format(pre)][index] = time_point
                    hdf5_file['{}_label'.format(pre)][index, ...] = data[None]
                    # hdf5_file['{}_prev_label'.format(pre)][index] = labels.index(prev_label)

                    sequence_ind += 1
            print('main loop took', timepackage.time() - begin_time)

        hdf5_file.close()
        with open(generated_vods_path, 'w') as f:
            for r in analyzed_vods:
                f.write('{}\n'.format(r))


def generate_data_for_cnn(rounds, vods, rounds_plus):
    # rounds = rounds[:2]
    generate_data(rounds)
    # generate_data_for_game_cnn(vods)


def save_round_info(rounds):
    with open(os.path.join(training_data_directory, 'rounds.txt'), 'w') as f:
        for r in rounds:
            f.write('{} {} {} {}\n'.format(r['id'], r['game']['match']['wl_id'], r['game']['game_number'],
                                           r['round_number']))


if __name__ == '__main__':
    rounds_plus = get_train_rounds_plus()
    rounds = get_train_rounds()
    for r in rounds:
        print(r['sequences'])
        local_path = get_vod_path(r['stream_vod'])
        if not os.path.exists(local_path):
            if get_local_path(r) is not None:
                shutil.move(get_local_path(r), local_path)
            else:
                print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
                get_local_file(r)
    save_round_info(rounds)

    vods = get_train_vods()
    # rounds = get_example_rounds()
    generate_data_for_cnn(rounds, vods, rounds_plus)
