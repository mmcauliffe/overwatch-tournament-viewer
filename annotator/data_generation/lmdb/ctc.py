import h5py
import os
import random
import cv2
import numpy as np

from annotator.data_generation.classes.base import DataGenerator, write_cache
from annotator.config import na_lab


class CTCDataGenerator(DataGenerator):
    def __init__(self, debug=False, map_size=10995116277):
        super(CTCDataGenerator, self).__init__(debug, map_size)
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
            assert s == self.label_set

    def save_label_set(self):
        path = os.path.join(self.training_directory, 'labels_set.txt')
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf8') as f:
                for c in self.label_set:
                    f.write('{}\n'.format(c))

    def add_new_round_info(self, r):
        super(CTCDataGenerator, self).add_new_round_info(r)
        if not self.generate_data:
            return

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