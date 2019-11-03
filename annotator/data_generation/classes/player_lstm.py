import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.player import PlayerStatusGenerator
from annotator.config import na_lab, sides, BOX_PARAMETERS, BASE_TIME_STEP
from annotator.utils import look_up_player_state
from annotator.api_requests import get_player_states
from annotator.game_values import HERO_SET, COLOR_SET, STATUS_SET, SPECTATOR_MODES


class PlayerLSTMGenerator(PlayerStatusGenerator):
    identifier = 'player_lstm'
    num_slots = 12
    num_variations = 1
    time_step = BASE_TIME_STEP
    frames_per_seq = 100
    num_train_slots = 10
    num_val_slots = 2

    def __init__(self):
        super(PlayerLSTMGenerator, self).__init__()
        self.images = {}
        self.data = {}
        self.current_sequence_index = 0
        self.process_index = 0
        self.y_offset = 0
        self.x_offset = 0

    def check_status(self, slot, time_point):
        for interval in self.status_state[slot]:
            if interval['end'] >= time_point >= interval['begin']:
                return True
        return False

    @property
    def minimum_time_step(self):
        return self.time_step

    def process_frame(self, frame, time_point, frame_ind):
        if not self.generate_data:
            return
        for slot in self.slots:
            d = self.lookup_data(slot, time_point)
            side = slot[0]
            zoomed = self.is_zoomed(time_point, side)
            if zoomed:
                params = self.zoomed_params[slot]
            else:
                params = self.slot_params[slot]
            x = params['x']
            y = params['y']
            if zoomed:
                box = frame[y + self.y_offset: y + self.zoomed_height + self.y_offset,
                      x + self.x_offset: x + self.zoomed_width + self.x_offset]
                box = cv2.resize(box, (self.image_height, self.image_width))
            else:
                box = frame[y + self.y_offset: y + self.image_height + self.y_offset,
                      x + self.x_offset: x + self.image_width + self.x_offset]
            if False:
                cv2.imshow('frame_{}'.format(slot), box)
                print(time_point)
                print(d)
                cv2.waitKey()
            box = np.transpose(box, axes=(2, 0, 1))
            self.images[slot][self.process_index, ...] = box[None]
            for k in self.sets.keys():
                self.data[slot][k][self.process_index] = self.sets[k].index(d[k])

        self.process_index += 1
        if self.process_index == self.frames_per_seq:
            self.save_current_sequence()

    def reset_cached_data(self):
        self.images = {}
        self.data = {}
        self.process_index = 0
        self.x_offset = random.randint(-3, 3)
        self.y_offset = random.randint(-3, 3)

        for slot in self.slots:
            self.images[slot] = np.zeros(
                (self.frames_per_seq, 3, int(self.image_height * self.resize_factor),
                 int(self.image_width * self.resize_factor)),
                dtype=np.uint8)
            self.data[slot] = {}
            for k in self.sets.keys():
                self.data[slot][k] = np.zeros((self.frames_per_seq,), dtype=np.uint8)

    def save_current_sequence(self):
        if not self.generate_data:
            return
        if self.process_index == 0:
            return
        for i, slot in enumerate(self.slots):
            if self.current_sequence_index >= len(self.indexes):
                continue
            index = self.indexes[self.current_sequence_index]
            if index < self.num_train:
                pre = 'train'
            else:
                pre = 'val'
                index -= self.num_train
            print(index, self.current_sequence_index, len(self.indexes))
            #for j in range(self.frames_per_seq):
            #    cv2.imshow('frame', np.transpose(self.images[slot][j, ...], (1, 2, 0)))
            #    for k in self.sets.keys():
            #        print(k, self.sets[k][self.data[slot][k][j]])
            #    cv2.waitKey()
            #    break
            self.hdf5_file["{}_img".format(pre)][index, ...] = self.images[slot][...]

            for k in self.sets.keys():
                self.hdf5_file['{}_{}_label'.format(pre, k)][index, ...] = self.data[slot][k][None]
            self.current_sequence_index += 1
        self.reset_cached_data()

    def add_new_round_info(self, r):
        if r['id'] <  9359:
            self.generate_data = False
            return
        self.current_round_id = r['id']
        self.hd5_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        if os.path.exists(self.hd5_path) or r['annotation_status'] not in self.usable_annotations:
            self.generate_data = False
            return
        self.get_data(r)

        expected_duration = 0
        for beg, end in r['sequences']:
            expected_duration += end - beg
            print(beg, end)
        print(expected_duration, expected_duration / self.time_step)
        per_slot_frames = int(expected_duration / self.time_step)
        print(per_slot_frames)
        if per_slot_frames % self.frames_per_seq == 0:
            num_seqs = int(per_slot_frames / self.frames_per_seq)
        else:
            num_seqs = int(per_slot_frames / self.frames_per_seq) + 1
        print(num_seqs)
        total_num_seqs = num_seqs * self.num_slots
        print(total_num_seqs)
        self.num_train = int(total_num_seqs * 0.8)
        self.num_val = total_num_seqs - self.num_train

        self.indexes = random.sample(range(total_num_seqs), total_num_seqs)

        self.analyzed_rounds.append(r['id'])
        self.current_round_id = r['id']
        self.generate_data = True
        self.figure_slot_params(r)

        train_shape = (self.num_train, self.frames_per_seq,
                       3, int(self.image_height *self.resize_factor), int(self.image_width*self.resize_factor))
        val_shape = (self.num_val, self.frames_per_seq,
                     3, int(self.image_height *self.resize_factor), int(self.image_width*self.resize_factor))
        self.hdf5_file = h5py.File(self.hd5_path, mode='w')

        for pre in ['train', 'val']:
            if pre == 'train':
                shape = train_shape
                count = self.num_train
            else:
                shape = val_shape
                count = self.num_val
            self.hdf5_file.create_dataset("{}_img".format(pre), shape, np.uint8)
            for k, s in self.sets.items():
                self.hdf5_file.create_dataset("{}_{}_label".format(pre, k), (count, self.frames_per_seq), np.uint8)

        self.process_index = 0
        self.current_sequence_index = 0
        self.left_color = r['game']['left_team']['color'].lower()
        self.right_color = r['game']['right_team']['color'].lower()
        self.spec_mode = r['spectator_mode'].lower()
        self.reset_cached_data()

    def lookup_data(self, slot, time_point):
        d = look_up_player_state(slot[0], slot[1], time_point, self.states, has_status=True)
        d['spectator_mode'] = self.spec_mode
        if slot[0] == 'left':
            d['color'] = self.left_color
        else:
            d['color'] = self.right_color
        d['side'] = slot[0]
        return d

    def cleanup_round(self):
        self.save_current_sequence()
        if not self.generate_data:
            return
        print('CLEAN UP')
        print(self.process_index, self.current_sequence_index, len(self.indexes))
        with open(self.rounds_analyzed_path, 'w') as f:
            for r in self.analyzed_rounds:
                f.write('{}\n'.format(r))
        self.hdf5_file.close()