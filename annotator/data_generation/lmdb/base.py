import os
import cv2
import lmdb
import random
import numpy as np
import pickle
from annotator.config import training_data_directory
from pympler import asizeof


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            #print(k)
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode('utf8')
            #print(asizeof.asizeof(k))
            #print(asizeof.asizeof(v))
            txn.put(k,v)


class DataGenerator(object):
    identifier = ''
    num_variations = 1
    time_step = 0.1
    num_slots = 1
    resize_factor = 1

    def __init__(self, debug=False, map_size=10995116277):
        self.training_directory = os.path.join(training_data_directory, self.identifier)
        os.makedirs(self.training_directory, exist_ok=True)
        self.generate_data = False
        self.states = None
        self.slots = []
        self.slot_params = {}
        self.sets = {}
        self.train_mean = None
        self.debug=debug
        if self.debug:
            os.makedirs(os.path.join(self.training_directory, 'debug', 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.training_directory, 'debug', 'val'), exist_ok=True)

    def instantiate_environment(self):
        # LMDB set up
        train_directory =  os.path.join(self.training_directory, 'training_set')
        first_run = not os.path.exists(train_directory)
        val_directory =  os.path.join(self.training_directory, 'val_set')
        self.train_env = lmdb.open(train_directory, map_size=self.train_map_size)
        self.val_env = lmdb.open(val_directory, map_size=self.val_map_size)

        if first_run:
            self.previous_train_count = 0
            self.previous_val_count = 0
            self.processed_rounds = set()
        else:
            with self.train_env.begin(write=False) as txn:
                self.previous_train_count = int(txn.get('num-samples'.encode('utf-8')))
                self.processed_rounds = set(txn.get('processed-rounds'.encode('utf8')).decode('utf8').split(','))
            with self.val_env.begin(write=False) as txn:
                self.previous_val_count = int(txn.get('num-samples'.encode('utf-8')))
                self.processed_rounds = set(txn.get('processed-rounds'.encode('utf8')).decode('utf8').split(','))
        self.train_cache = {}
        self.val_cache = {}

    def check_set_info(self):
        for k in self.sets.keys():
            path = os.path.join(self.training_directory, '{}_set.txt'.format(k))
            s = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf8') as f:
                    for line in f:
                        s.append(line.strip())
                assert s == self.sets[k]

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

    def get_data(self, r):
        pass

    def figure_slot_params(self, r):
        pass

    def process_frame(self, frame, time_point):
        if not self.generate_data:
            return
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
                box = frame[y + y_offset: y + self.image_height + y_offset,
                      x + x_offset: x + self.image_width + x_offset]
                if self.resize_factor != 1:
                    box = cv2.resize(box, (0, 0), fx=self.resize_factor, fy=self.resize_factor)

                box = np.transpose(box, axes=(2, 0, 1))
                train_check = random.random() <= 0.8
                if train_check:
                    pre = 'train'
                    index = self.previous_train_count
                    image_key = 'image-%09d' % index
                    round_key = 'round-%09d' % index
                    time_point_key = 'time_point-%09d' % index
                    self.train_cache[image_key] = pickle.dumps(box)
                    self.train_cache[round_key] = str(self.current_round_id)
                    self.train_cache[time_point_key] = '{:.1f}'.format(time_point)
                    for k in self.sets.keys():
                        key = '%s-%09d'% (k, index)
                        self.train_cache[key] = d[k]
                    self.previous_train_count += 1
                else:
                    pre = 'val'
                    index = self.previous_val_count
                    image_key = 'image-%09d' % index
                    round_key = 'round-%09d' % index
                    time_point_key = 'time_point-%09d' % index
                    self.val_cache[image_key] = pickle.dumps(box)
                    self.val_cache[round_key] = str(self.current_round_id)
                    self.val_cache[time_point_key] = '{:.1f}'.format(time_point)
                    for k in self.sets.keys():
                        key = '%s-%09d'% (k, index)
                        self.val_cache[key] = d[k]
                    self.previous_val_count += 1

                self.process_index += 1
                if self.process_index % 1000 == 0:
                    write_cache(self.train_env, self.train_cache)
                    self.train_cache = {}
                    write_cache(self.val_env, self.val_cache)
                    self.val_cache = {}
                if self.debug:

                    filename = '{}_{}.jpg'.format(' '.join(d.values()), index).replace(':', '')
                    cv2.imwrite(os.path.join(self.training_directory, 'debug', pre,
                                         filename), np.transpose(box, axes=(1,2,0)))

    def add_new_round_info(self, r):
        if str(r['id']) in self.processed_rounds:
            self.generate_data = False
            return
        self.current_round_id = r['id']
        self.generate_data = True
        self.figure_slot_params(r)
        self.processed_rounds.add(str(r['id']))
        self.get_data(r)

        self.process_index = 0
        self.train_cache = {}
        self.val_cache = {}

    def cleanup_round(self):
        if not self.generate_data:
            return
        self.train_cache['num-samples'] = str(self.previous_train_count - 1)
        self.train_cache['processed-rounds'] = ','.join(sorted(self.processed_rounds))
        if self.train_cache:
            write_cache(self.train_env, self.train_cache)
            self.train_cache = {}
        self.val_cache['num-samples'] = str(self.previous_val_count - 1)
        self.val_cache['processed-rounds'] = ','.join(sorted(self.processed_rounds))
        if self.val_cache:
            write_cache(self.val_env, self.val_cache)
            self.val_cache = {}

    def cleanup(self):
        self.train_env.close()
        self.val_env.close()

    def calculate_map_size(self, rounds):
        pass
