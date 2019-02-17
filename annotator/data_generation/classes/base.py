import os
import cv2

from annotator.config import training_data_directory


class DataGenerator(object):
    identifier = ''
    num_variations = 1
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
        self.data_directory = os.path.join(self.training_directory, 'data')
        self.hd5_path = None
        self.generate_data = False
        self.hdf5_file = None
        self.states = None
        self.slots = []
        self.slot_params = {}
        self.sets = {}
        self.end_sets = {}
        self.immutable_sets = {}
        self.train_mean = None

    def check_set_info(self):
        for k in self.sets.keys():
            path = os.path.join(self.training_directory, '{}_set.txt'.format(k))
            s = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf8') as f:
                    for line in f:
                        s.append(line.strip())
                assert s == self.sets[k]
        for k in self.end_sets.keys():
            path = os.path.join(self.training_directory, '{}_set.txt'.format(k))
            s = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf8') as f:
                    for line in f:
                        s.append(line.strip())
                assert s == self.end_sets[k]
        for k in self.immutable_sets.keys():
            path = os.path.join(self.training_directory, '{}_set.txt'.format(k))
            s = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf8') as f:
                    for line in f:
                        s.append(line.strip())
                print(s)
                print(self.immutable_sets[k])
                assert s == self.immutable_sets[k]

    def display_current_frame(self, frame, time_point):
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
        for k, s in self.end_sets.items():
            path = os.path.join(self.training_directory, '{}_set.txt'.format(k))
            if not os.path.exists(path):
                with open(path, 'w', encoding='utf8') as f:
                    for p in s:
                        f.write('{}\n'.format(p))
        for k, s in self.immutable_sets.items():
            path = os.path.join(self.training_directory, '{}_set.txt'.format(k))
            if not os.path.exists(path):
                with open(path, 'w', encoding='utf8') as f:
                    for p in s:
                        f.write('{}\n'.format(p))

    def get_data(self, r):
        pass

    def figure_slot_params(self, r):
        pass

    def cleanup_round(self):
        if not self.generate_data:
            return
        with open(self.rounds_analyzed_path, 'w') as f:
            for r in self.analyzed_rounds:
                f.write('{}\n'.format(r))
        #self.hdf5_file["train_mean"][...] = self.train_mean
        self.hdf5_file.close()
