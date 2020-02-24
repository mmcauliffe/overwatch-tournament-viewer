import os
import cv2
import numpy as np
import random
import h5py
from annotator.config import training_data_directory, BASE_TIME_STEP


class DataGenerator(object):
    identifier = ''
    num_variations = 1
    time_step = BASE_TIME_STEP
    num_slots = 1
    resize_factor = 1
    usable_annotations = ['M']

    def __init__(self, debug=False):
        self.debug = debug
        self.training_directory = os.path.join(training_data_directory, self.identifier)
        os.makedirs(self.training_directory, exist_ok=True)
        self.rounds_analyzed_path = os.path.join(self.training_directory, 'rounds.txt')
        self.analyzed_rounds = []
        if os.path.exists(self.rounds_analyzed_path):
            with open(self.rounds_analyzed_path, 'r') as f:
                for line in f:
                    self.analyzed_rounds.append(int(line.strip()))
        self.data_directory = os.path.join(self.training_directory, 'data')
        self.hd5_path = None
        self.generate_data = False
        self.hdf5_file = None
        self.states = None
        self.slots = []
        self.slot_params = {}
        self.sets = {}
        if debug:
            os.makedirs(os.path.join(self.training_directory, 'debug', 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.training_directory, 'debug', 'val'), exist_ok=True)

    def check_set_info(self):
        return
        for k in self.sets.keys():
            path = os.path.join(self.training_directory, '{}_set.txt'.format(k))
            s = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf8') as f:
                    for line in f:
                        s.append(line.strip())
                assert s == self.sets[k][len(s)]

    def display_current_frame(self, frame, time_point, frame_ind):
        for slot in self.slots:
            if isinstance(slot, (list, tuple)):
                slot_name = '_'.join(map(str, slot))
            else:
                slot_name = slot
            params = self.slot_params[slot]
            x = params['x']
            y = params['y']
            box = frame[y: y + self.image_height,
                  x: x + self.image_width]
            cv2.imshow('{}_{}'.format(self.identifier, slot_name), box)

    def save_set_info(self):
        for k, s in self.sets.items():
            path = os.path.join(self.training_directory, '{}_set.txt'.format(k))
            if not os.path.exists(path):
                with open(path, 'w', encoding='utf8') as f:
                    for p in s:
                        f.write('{}\n'.format(p))

    def get_data(self, r):
        pass

    def figure_slot_params(self, r):
        pass

    def process_frame(self, frame, time_point, frame_ind):
        if not self.generate_data:
            return
        frame = frame['frame']
        for slot in self.slots:
            d = self.lookup_data(slot, time_point)
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
                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                if self.resize_factor != 1:
                    box = cv2.resize(box, (0, 0), fx=self.resize_factor, fy=self.resize_factor)

                box = np.transpose(box, axes=(2, 0, 1))
                self.data["{}_img".format(pre)][index, ...] = box[None]
                self.data["{}_round".format(pre)][index] = self.current_round_id
                self.data["{}_time_point".format(pre)][index] = time_point

                for k, s in self.sets.items():
                    self.data["{}_{}_label".format(pre, k)][index] = s.index(d[k])

                self.process_index += 1
                if self.debug:

                    filename = '{}_{}.jpg'.format(' '.join(d.values()), index).replace(':', '')
                    cv2.imwrite(os.path.join(self.training_directory, 'debug', pre,
                                         filename), np.transpose(box, axes=(1,2,0)))

    def cleanup_round(self):
        if not self.generate_data:
            return
        with h5py.File(self.hd5_path, mode='w') as hdf5_file:
            for k, v in self.data.items():
                hdf5_file.create_dataset(k, v.shape, v.dtype)
                hdf5_file[k][:] = v[:]
        with open(self.rounds_analyzed_path, 'w') as f:
            for r in self.analyzed_rounds:
                f.write('{}\n'.format(r))

    def add_new_round_info(self, r, reset = False):
        import shutil
        spec_dir = os.path.join(self.training_directory, r['spectator_mode'].lower())
        os.makedirs(spec_dir, exist_ok=True)
        old_path = os.path.join(self.training_directory, '{}.hdf5'.format(r['id']))
        self.hd5_path = os.path.join(spec_dir, '{}.hdf5'.format(r['id']))
        if os.path.exists(old_path):
            shutil.move(old_path, self.hd5_path)
            self.generate_data = False
            return
        self.current_round_id = r['id']
        if reset and os.path.exists(self.hd5_path):
            os.remove(self.hd5_path)
        if os.path.exists(self.hd5_path) or r['annotation_status'] not in self.usable_annotations:
            self.generate_data = False
            return
        self.get_data(r)

        num_frames = 0
        for beg, end in r['sequences']:
            beg += 0.1
            beg = round(beg, 1)
            end -= 0.1
            end = round(end, 1)
            expected_duration = end - beg
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
        print(self.image_height, int(self.image_height *self.resize_factor))
        train_shape = (self.num_train, 3, int(self.image_height *self.resize_factor), int(self.image_width*self.resize_factor))
        val_shape = (self.num_val, 3, int(self.image_height *self.resize_factor), int(self.image_width*self.resize_factor))

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
            for k, s in self.sets.items():
                self.data["{}_{}_label".format(pre, k)] =  np.zeros((count,), dtype=np.uint8)

        self.process_index = 0
