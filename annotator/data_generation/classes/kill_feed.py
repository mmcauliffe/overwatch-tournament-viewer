import h5py
import os
import random
import cv2
import heapq
import numpy as np
import torch

from annotator.data_generation.classes.ctc import CTCDataGenerator
from annotator.game_values import COLOR_SET, HERO_SET, LABEL_SET, SPECTATOR_MODES
from annotator.config import BOX_PARAMETERS, na_lab, BASE_TIME_STEP
from annotator.api_requests import get_kf_events
from annotator.utils import get_event_ranges
from annotator.training.ctc_helper import loadData


def check_is_npc(event):
    return event['second_hero'] not in HERO_SET + ['b.o.b._npc', 'b.o.b.']


def construct_kf_at_time(events, time, use_raw):
    if use_raw:
        window = 7.3
        k = 6
    else:
        window = 9
        k = 7
    possible_kf = []
    beyond = []
    for e in events:
        if e['time_point'] > time + 0.5:
            break
        elif e['time_point'] > time:
            beyond.append(e)
        if round(time - window, 1) <= e['time_point'] <= time:
            e['is_npc'] = check_is_npc(e)
            possible_kf.append(e)
    possible_kf = possible_kf[-k:]
    for e in beyond:
        if e['time_point'] <= round(time + 0.2, 1):
            for e2 in beyond:
                if e == e2:
                    continue
                if round(e2['time_point'] - e['time_point'], 1) <= 0.2:
                    extra_check = False
                    for e3 in beyond:
                        if e3 == e or e3 == e2:
                            continue
                        if round(e3['time_point'] - e2['time_point'], 1) <= 0.2:
                            extra_check = True
                            break
                    if not extra_check:
                        break
            else:
                e['is_npc'] = check_is_npc(e)
                possible_kf.append(e)
    possible_kf.reverse()
    return possible_kf


def is_same_event(e_one, e_two, fields):
    #if e_one['first_hero'] == 'd.va' == e_two['first_hero'] and e_one['first_side'] == e_two['first_side'] \
    #        and e_one['ability'] == 'defense matrix' == e_two['ability']:
    #    return True
    for f in fields:
        if e_one[f] != e_two[f]:
            return False
    return True


def find_best_event_match(kf, slot_data, is_npc):
    fields = ['second_hero', 'second_side', 'first_hero', 'first_side', 'ability', 'headshot', 'environmental', 'assisting_heroes']
    ordering = [-1, -2, -3, -5]
    for e in kf:
        if is_same_event(e, slot_data, fields) and is_npc == check_is_npc(e) == check_is_npc(slot_data):
            return e
    for o in ordering:
        for e in kf:
            if is_same_event(e, slot_data, fields[:o]) and is_npc == check_is_npc(e) == check_is_npc(slot_data):
                return e
    return None


class KillFeedCTCGenerator(CTCDataGenerator):
    identifier = 'kill_feed_ctc'
    num_slots = 6
    num_variations = 1
    time_step = round(BASE_TIME_STEP * 2, 1)
    usable_annotations = ['M', 'O']

    def __init__(self, debug=False, exists_model_directory=None, kf_ctc_model_directory=None):
        self.use_raw = exists_model_directory is None
        if self.use_raw:
            self.usable_annotations = ['M']
            self.identifier = 'kill_feed_ctc_base'
        super(KillFeedCTCGenerator, self).__init__()
        self.exists_model = None
        self.model = None
        self.exists_model_directory = exists_model_directory
        self.kf_ctc_model_directory = kf_ctc_model_directory
        self.image_width = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['WIDTH']
        self.image_height = BOX_PARAMETERS['O']['KILL_FEED_SLOT']['HEIGHT']
        if exists_model_directory:
            from torch.autograd import Variable
            from annotator.models.cnn import KillFeedCNN
            from annotator.models.crnn import SideKillFeedCRNN
            from annotator.training.helper import load_set
            set_paths = {
                'exist': os.path.join(exists_model_directory, 'exist_set.txt'),
                'size': os.path.join(exists_model_directory, 'size_set.txt'),
            }

            input_set_paths = {
                #'spectator_mode': os.path.join(exists_model_directory, 'spectator_mode_set.txt'),
            }

            sets = {}
            input_sets = {}
            for k, v in set_paths.items():
                sets[k] = load_set(v)
            for k, v in input_set_paths.items():
                input_sets[k] = load_set(v)
            self.exists_model = KillFeedCNN(sets, input_sets=input_sets)
            self.exists_model.eval()
            for p in self.exists_model.parameters():
                p.requires_grad = False
            self.exists_model.cuda()
            self.images = Variable(torch.FloatTensor(1, 3,
                                                     int(self.image_height * self.resize_factor),
                                                     int(self.image_width * self.resize_factor)).cuda())
            self.left_colors = Variable(torch.FloatTensor(6, 3).cuda())
            self.right_colors = Variable(torch.FloatTensor(6, 3).cuda())
        if kf_ctc_model_directory:
            label_set = load_set(os.path.join(kf_ctc_model_directory, 'labels_set.txt'))

            self.model = SideKillFeedCRNN(label_set)
            self.model.load_state_dict(torch.load(os.path.join(kf_ctc_model_directory, 'model.pth')))
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.cuda()
        self.label_set = LABEL_SET
        self.exist_sets = {
            'exist': ['empty', 'not_empty'],
            'size': ['full', 'half']
        }
        self.save_label_set()
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
        for name, s in self.exist_sets.items():
            path = os.path.join(self.training_directory, '{}_set.txt'.format(name))
            with open(path, 'w', encoding='utf8') as f:
                for c in s:
                    f.write('{}\n'.format(c))

    def figure_slot_params(self, r):
        from datetime import datetime
        self.slot_params = {}
        film_format = r['stream_vod']['film_format']['code']
        params = BOX_PARAMETERS[film_format]['KILL_FEED_SLOT']
        broadcast_date = datetime.strptime(r['stream_vod']['broadcast_date'], '%Y-%m-%dT%H:%M:%SZ')
        status_date_start = datetime(2019, 1, 1)
        self.half_size_npcs = broadcast_date > status_date_start
        self.margin = params['MARGIN']
        for s in self.slots:
            self.slot_params[s] = {}
            self.slot_params[s]['x'] = params['X']
            self.slot_params[s]['y'] = params['Y'] + (params['HEIGHT'] + params['MARGIN']) * (s)

    def lookup_data(self, kf, exists, pred_kf, slot, time_point):
        if slot > len(kf) - 1:
            is_npc = False
        else:
            is_npc = kf[slot]['is_npc']
        if exists['exist']:
            exist = exists['exist'][slot]
            size = exists['size'][slot]
            is_npc = size == 'half'
            if exist != 'empty' and pred_kf and slot in pred_kf:
                d = find_best_event_match(kf, pred_kf[slot], is_npc)
            else:
                if exist == 'empty':
                    d = None
                elif slot > len(kf) - 1:
                    d = None
                else:
                    d = kf[slot]
                    if d['time_point'] > time_point:
                        d = None
        else:
            if slot > len(kf) - 1:
                d = None
            else:
                d = kf[slot]
                if d['time_point'] > time_point:
                    d = None

        if d is None or d['second_hero'] == 'n/a':
            sequence, raw_sequence = [], []
        else:
            raw_sequence = []
            if d['headshot'] and not d['ability'].endswith('headshot'):
                d['ability'] += ' headshot'
            if d['first_hero'] == 'n/a':
                pass
            else:
                #first_color = d['first_color']
                #if self.spec_mode != 'original':
                #    if first_color != 'white':
                #        first_color = 'nonwhite'
                raw_sequence.append(d['first_side'])
                raw_sequence.append(d['first_hero'])
                if d['assisting_heroes']:
                    for h in d['assisting_heroes']:
                        raw_sequence.append(h.lower() + '_assist')
            if d['ability'] == 'n/a':
                d['ability'] = 'primary'
            raw_sequence.append(d['ability'])
            if d['environmental']:
                raw_sequence.append('environmental')
            second = d['second_hero']
            if second not in HERO_SET and second+'_npc' in LABEL_SET:
                second += '_npc'
            raw_sequence.append(second)
            #second_color = d['second_color']
            #if self.spec_mode != 'original':
            #    if second_color != 'white':
            #        second_color = 'nonwhite'
            raw_sequence.append(d['second_side'])
            raw_sequence = [x for x in raw_sequence]
            sequence = [self.label_set.index(x) for x in raw_sequence]
        return sequence, raw_sequence, is_npc

    def display_current_frame(self, frame, time_point, frame_ind):
        shift = 0
        kf = construct_kf_at_time(self.states, time_point, self.use_raw)
        exists = {'exist': [], 'size': []}
        images = None
        left_colors = []
        right_colors = []
        if self.exists_model:
            for slot in self.slots:
                params = self.slot_params[slot]
                x = params['x']
                y = params['y']

                #if self.half_size_npcs and is_npc:
                #    y -= int(self.image_height / 4) - 2
                box = frame[y - shift: y + self.image_height - shift,
                      x: x + self.image_width]
                #cv2.imshow('cur_slot', box)
                #cv2.waitKey()
                image = torch.from_numpy(np.transpose(box, axes=(2, 0, 1))[None]).float().cuda()
                image = ((image / 255) - 0.5) / 0.5
                #print('load exist image', time.time()-b)
                #b = time.time()
                with torch.no_grad():
                    predicteds = self.exists_model({'image': image,
                                                    })
                    for k, s in self.exists_model.sets.items():
                        _, predicteds[k] = torch.max(predicteds[k], 1)
                        exists[k].append(s[predicteds[k][0]])
                print(slot, exists)
                if exists['size'][-1] == 'half':
                    #show = True
                    shift += int(self.image_height/2)
                if exists['exist'][-1] == 'empty':
                    continue
                if self.model:
                    if images is None:
                        images = image
                    else:
                        images = torch.cat((images, image), 0)
                    left_colors.append(self.left_color_hex)
                    right_colors.append(self.right_color_hex)

        if self.exists_model and self.model and images is not None:
            with torch.no_grad():
                left_colors = (((torch.FloatTensor(left_colors) / 255) - 0.5) / 0.5).cuda()
                right_colors = (((torch.FloatTensor(right_colors) / 255) - 0.5) / 0.5).cuda()
                loadData(self.left_colors, left_colors)
                loadData(self.right_colors, right_colors)
                cur_kf = self.model.parse_image(images, self.left_colors, self.right_colors)
            slot_ind = 0
            pred_kf = {}
            for slot in self.slots:
                if exists['exist'][slot] != 'empty':
                    pred_kf[slot] = cur_kf[slot_ind]
                    slot_ind += 1
        else:
            pred_kf = {}
        print('EXISTS', exists)
        print('PRED', pred_kf)
        print('MANUAL', kf)
        shift = 0
        for slot in self.slots:
            if isinstance(slot, (list, tuple)):
                slot_name = '_'.join(map(str, slot))
            else:
                slot_name = slot
            params = self.slot_params[slot]
            x = params['x']
            y = params['y']
            #if self.half_size_npcs and is_npc:
            #    y -= int(self.image_height /4) - 2
            box = frame[y - shift: y + self.image_height - shift,
                  x: x + self.image_width]
            sequence, raw_sequence, is_npc = self.lookup_data(kf, exists, pred_kf, slot, time_point)
            print(slot, is_npc)
            print(sequence)
            print(raw_sequence)
            if self.half_size_npcs and is_npc:
                if sequence:
                    label = 'half_sized'
                else:
                    label = 'half_sized_empty'
            else:
                if sequence:
                    label = 'full_sized'
                else:
                    label = 'empty'
            print(label)

            cv2.imshow('{}_{}'.format(self.identifier, slot_name), box)
            if self.half_size_npcs and is_npc:
                shift += int(self.image_height /2) - int(self.margin)

    def process_frame(self, frame, time_point, frame_ind):
        frame = frame['frame']
        if not self.generate_data:
            return
        for rd in self.ranges:
            if rd['begin'] <= time_point <= rd['end']:
                break
        else:
            return

        kf = construct_kf_at_time(self.states, time_point, self.use_raw)
        shift = 0
        exists = {'exist': [], 'size': []}
        images = None
        left_colors = []
        right_colors = []
        if self.exists_model:
            for slot in self.slots:
                params = self.slot_params[slot]
                x = params['x']
                y = params['y']

                box = frame[y - shift: y + self.image_height - shift,
                      x: x + self.image_width]
                image = torch.from_numpy(np.transpose(box, axes=(2, 0, 1))[None]).float().cuda()
                image = ((image / 255) - 0.5) / 0.5
                with torch.no_grad():
                    predicteds = self.exists_model({'image': image,
                                                    })
                    for k, s in self.exists_model.sets.items():
                        _, predicteds[k] = torch.max(predicteds[k], 1)
                        exists[k].append(s[predicteds[k][0]])
                if exists['exist'][-1] == 'empty':
                    continue
                if self.model:
                    if images is None:
                        images = image
                    else:
                        images = torch.cat((images, image), 0)
                    left_colors.append(self.left_color_hex)
                    right_colors.append(self.right_color_hex)
                if exists['size'][-1] == 'half':
                    #show = True
                    shift += int(self.image_height/2)

        if self.exists_model and self.model and images is not None:
            with torch.no_grad():
                left_colors = (((torch.FloatTensor(left_colors) / 255) - 0.5) / 0.5).cuda()
                right_colors = (((torch.FloatTensor(right_colors) / 255) - 0.5) / 0.5).cuda()
                loadData(self.left_colors, left_colors)
                loadData(self.right_colors, right_colors)
                cur_kf = self.model.parse_image(images, self.left_colors, self.right_colors)
            slot_ind = 0
            pred_kf = {}
            for slot in self.slots:
                if exists['exist'][slot] != 'empty':
                    pred_kf[slot] = cur_kf[slot_ind]
                    slot_ind += 1
        else:
            pred_kf = {}

        shift = 0
        for slot in self.slots:
            params = self.slot_params[slot]

            variation_set = []
            while len(variation_set) < self.num_variations:
                x_offset = random.randint(-1, 1)
                y_offset = random.randint(-3, 3)
                if (x_offset, y_offset) in variation_set:
                    continue
                variation_set.append((x_offset, y_offset))

            x = params['x']
            y = params['y']

            sequence, raw_sequence, is_npc = self.lookup_data(kf, exists, pred_kf, slot, time_point)

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
                d = {'size': 'full', 'exist': 'empty'}
                if self.half_size_npcs and is_npc:
                    d['size'] = 'half'
                if sequence:
                    d['exist'] = 'not_empty'

                self.data["{}_img".format(pre)][index, ...] = box[None]
                self.data["{}_round".format(pre)][index] = self.current_round_id
                self.data["{}_time_point".format(pre)][index] = time_point
                # self.train_mean += box / self.hdf5_file['train_img'].shape[0]
                sequence_length = len(sequence)
                #if time_point > 262:
                #    print(sequence)
                #    print(raw_sequence)
                #    cv2.imshow('frame', np.transpose(box, (1, 2, 0)))
                #    cv2.waitKey()
                if sequence:
                    self.data["{}_left_color".format(pre)][index] = self.left_color_hex
                    self.data["{}_right_color".format(pre)][index] = self.right_color_hex
                    self.data["{}_label_sequence_length".format(pre)][index] = sequence_length
                    self.data["{}_label_sequence".format(pre)][index, 0:len(sequence)] = sequence
                else:
                    self.ignored_indexes.append(index)
                for name, s in self.exist_sets.items():
                    self.data["{}_{}_label".format(pre, name)][index] = s.index(d[name])
                self.process_index += 1
            if self.half_size_npcs and is_npc:
                shift += int(self.image_height /2) - int(self.margin)
        if frame_ind % 10 == 0:
            self.states = [x for x in self.states if x['time_point'] > time_point - 8]

    def get_data(self, r):
        self.spec_mode = r['spectator_mode'].lower()
        self.left_color = r['game']['left_team']['color'].lower()
        self.right_color = r['game']['right_team']['color'].lower()
        self.left_color_hex = r['game']['left_team_color_hex'].lstrip('#')
        self.right_color_hex = r['game']['right_team_color_hex'].lstrip('#')
        self.left_color_hex = tuple(int(self.left_color_hex[i:i+2], 16) for i in (0, 2, 4))
        self.right_color_hex = tuple(int(self.right_color_hex[i:i+2], 16) for i in (0, 2, 4))
        self.states = get_kf_events(r['id'])

    def set_up_models(self):
        if self.exists_model_directory:
            spec_dir = os.path.join(self.exists_model_directory, self.spectator_mode)
            if os.path.exists(spec_dir):
                exists_model_path = os.path.join(spec_dir, 'model.pth')
                print('Using {} exists model!'.format(self.spectator_mode))
            else:
                exists_model_path = os.path.join(self.exists_model_directory, 'model.pth')
                print('Using base exists model!')
            self.exists_model.load_state_dict(torch.load(exists_model_path))
        if self.kf_ctc_model_directory:
            spec_dir = os.path.join(self.kf_ctc_model_directory, self.spectator_mode)
            if os.path.exists(spec_dir):
                model_path = os.path.join(spec_dir, 'model.pth')
                print('Using {} kf slot model!'.format(self.spectator_mode))
            else:
                model_path = os.path.join(self.kf_ctc_model_directory, 'model.pth')
                print('Using base kf slot model!')
            self.model.load_state_dict(torch.load(model_path))

    def add_new_round_info(self, r, reset=False):
        self.current_round_id = r['id']
        self.spectator_mode = r['spectator_mode'].lower()
        spec_mode_directory = os.path.join(self.training_directory, self.spectator_mode)
        os.makedirs(spec_mode_directory, exist_ok=True)
        self.hd5_path = os.path.join(spec_mode_directory, '{}.hdf5'.format(r['id']))
        self.exist_hd5_path = os.path.join(spec_mode_directory, '{}_exists.hdf5'.format(r['id']))
        if reset and os.path.exists(self.hd5_path):
            os.remove(self.hd5_path)
        if os.path.exists(self.hd5_path) or r['annotation_status'] not in self.usable_annotations:
            self.generate_data = False
            return
        self.set_up_models()
        self.time_step = round(BASE_TIME_STEP * 3, 1)
        if r['annotation_status'] == 'O':
            self.time_step = round(self.time_step * 2, 1)
        self.get_data(r)
        if not self.states:
            self.generate_data = False
            return
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
        self.ignored_indexes = []
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
            self.data['{}_left_color'.format(pre)] = np.zeros((count,3), dtype=np.uint8)
            self.data['{}_right_color'.format(pre)] = np.zeros((count,3), dtype=np.uint8)
            for name in self.exist_sets.keys():
                self.data["{}_{}_label".format(pre, name)] = np.zeros((count,), dtype=np.uint8)
            self.data["{}_label_sequence".format(pre)] = np.zeros((count, self.max_sequence_length),
                                          dtype=np.int16)
            self.data["{}_label_sequence".format(pre)][:, :] = len(self.label_set)
            self.data["{}_label_sequence_length".format(pre)] = np.ones((count,), dtype=np.uint8)

        self.process_index = 0

    def cleanup_round(self):
        if not self.generate_data:
            return
        with h5py.File(self.exist_hd5_path, mode='w') as hdf5_file:
            for k, v in self.data.items():
                if 'label_sequence' in k:
                    continue
                if 'color' in k:
                    continue
                hdf5_file.create_dataset(k, v.shape, v.dtype)
                hdf5_file[k][:] = v[:]
        if self.ignored_indexes:
            for k, v in self.data.items():
                self.data[k] = np.delete(v, self.ignored_indexes, axis=0)
        with h5py.File(self.hd5_path, mode='w') as hdf5_file:
            for k, v in self.data.items():
                skip = False
                for name in self.exist_sets.keys():

                    if name in k:
                        skip = True
                        break
                if skip:
                    continue
                hdf5_file.create_dataset(k, v.shape, v.dtype)
                hdf5_file[k][:] = v[:]
        with open(self.rounds_analyzed_path, 'w') as f:
            for r in self.analyzed_rounds:
                f.write('{}\n'.format(r))