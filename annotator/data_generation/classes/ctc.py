import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.base import DataGenerator
from annotator.config import na_lab


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
        self.process_index = 0

    def check_set_info(self):
        path = os.path.join(self.training_directory, 'labels_set.txt')
        if os.path.exists(path):
            s = []
            with open(path, 'r', encoding='utf8') as f:
                for line in f:
                    s.append(line.strip())
            print(s, self.label_set)
            #assert s == self.label_set

    def save_label_set(self):
        path = os.path.join(self.training_directory, 'labels_set.txt')
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf8') as f:
                for c in self.label_set:
                    f.write('{}\n'.format(c))

    def add_new_round_info(self, r):
        self.current_round_id = r['id']
        self.hd5_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        if os.path.exists(self.hd5_path) or r['annotation_status'] not in self.usable_annotations:
            self.generate_data = False
            return
        num_frames = 0
        for beg, end in r['sequences']:
            print(beg, end)
            expected_frame_count = int((end - beg) / self.time_step)
            num_frames += (int(expected_frame_count) + 1) * self.num_slots

        num_frames *= self.num_variations
        self.num_train = int(num_frames * 0.8)
        self.num_val = num_frames - self.num_train
        self.generate_data = True
        self.figure_slot_params(r)
        self.analyzed_rounds.append(r['id'])

        self.indexes = random.sample(range(num_frames), num_frames)

        train_shape = (
            self.num_train, 3, int(self.image_width * self.resize_factor), int(self.image_height * self.resize_factor))
        val_shape = (
            self.num_val, 3, int(self.image_width * self.resize_factor), int(self.image_height * self.resize_factor))
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
            self.hdf5_file.create_dataset("{}_label_sequence".format(pre), (count, self.max_sequence_length),
                                          np.int16, maxshape=(None, self.max_sequence_length),
                                          fillvalue=len(self.label_set))
            self.hdf5_file.create_dataset("{}_label_sequence_length".format(pre), (count,), np.uint8,
                                          maxshape=(None,), fillvalue=1)
            self.hdf5_file.create_dataset("{}_spectator_mode".format(pre), (count,), np.uint8, maxshape=(None,))

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
                else:
                    pre = 'val'
                    index -= self.num_train
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
                bw = np.swapaxes(bw, 1, 0)
                self.hdf5_file["{}_img".format(pre)][index, ...] = bw[None]
                #self.train_mean += bw / self.hdf5_file['train_img'].shape[0]
                sequence_length = len(sequence)
                if sequence:
                    self.hdf5_file["{}_label_sequence_length".format(pre)][index] = sequence_length
                    self.hdf5_file["{}_label_sequence".format(pre)][index, 0:len(sequence)] = sequence
                self.process_index += 1
        #cv2.waitKey(0)
