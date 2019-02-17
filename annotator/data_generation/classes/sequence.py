import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.base import DataGenerator


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
        self.current_round_id = None

    def add_new_round_info(self, r):
        self.current_round_id = r['id']
        self.hd5_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        if os.path.exists(self.hd5_path):
            self.generate_data = False
            return
        self.num_sequences = 0
        for beg, end in r['sequences']:
            print(beg, end)
            expected_frame_count = int((end - beg) / self.time_step)
            self.num_sequences += (int(expected_frame_count / self.frames_per_seq) + 1) * self.num_slots
        self.current_sequence_index = 0
        self.num_sequences *= self.num_variations
        self.num_train = int(self.num_sequences * 0.8)
        self.num_val = self.num_sequences - self.num_train
        self.analyzed_rounds.append(r['id'])
        self.generate_data = True
        self.figure_slot_params(r)
        self.indexes = random.sample(range(self.num_sequences), self.num_sequences)
        train_shape = (self.num_train, self.frames_per_seq, int(self.image_height * self.resize_factor),
                       int(self.image_width * self.resize_factor), 3)
        val_shape = (self.num_val, self.frames_per_seq, int(self.image_height * self.resize_factor),
                     int(self.image_width * self.resize_factor), 3)

        self.hdf5_file = h5py.File(self.hd5_path, mode='w')
        #self.train_mean = np.zeros(train_shape[1:], np.float32)
        #self.hdf5_file.create_dataset("train_mean", train_shape[2:], np.float32)
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
                #self.train_mean += box / (self.hdf5_file['train_img'].shape[0]*self.hdf5_file['train_img'].shape[1])
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
            self.variation_sets[slot] = []
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
                else:
                    pre = 'val'
                    index -= self.num_train
                self.hdf5_file["{}_img".format(pre)][index, ...] = self.images[slot][i, ...]
                self.hdf5_file["{}_round".format(pre)][index] = self.current_round_id
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
        #self.hdf5_file["train_mean"][...] = self.train_mean
        self.hdf5_file.close()
