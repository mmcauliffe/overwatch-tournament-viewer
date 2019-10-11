import os
import torch
import numpy as np
import cv2
from annotator.annotator.classes.base import BaseAnnotator
from torch.autograd import Variable

from annotator.models.crnn import PlayerNameCRNN, KillFeedCRNN
from annotator.training.helper import load_set
from annotator.training.ctc_helper import loadData
from collections import defaultdict, Counter
from annotator.config import sides, BOX_PARAMETERS


class PlayerNameAnnotator(BaseAnnotator):
    resize_factor = 1
    time_step = 5
    batch_size = 100
    identifier = 'player_name'
    box_settings = 'LEFT_NAME'

    def __init__(self, film_format, model_directory, device, spectator_mode, debug=False):
        super(PlayerNameAnnotator, self).__init__(film_format, device, debug)
        self.spectator_mode = spectator_mode
        self.figure_slot_params(film_format)
        self.model_directory = model_directory
        label_set = load_set(os.path.join(model_directory, 'labels_set.txt'))
        spectator_mode_set = load_set(os.path.join(model_directory, 'spectator_mode_set.txt'))
        self.model = KillFeedCRNN(label_set, spectator_mode_set)
        self.model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pth')))
        self.model.eval()
        self.model.to(device)
        self.to_predict = {}
        self.names = {}
        self.image_height = self.params['HEIGHT'] * 2
        self.image_width = self.params['WIDTH']
        self.shape = (self.batch_size, 3, self.image_height, self.image_width)
        self.images = Variable(torch.FloatTensor(self.batch_size, 3, self.image_height, self.image_width).to(device))
        self.spectator_mode_input = Variable(torch.LongTensor(self.batch_size).to(device))
        self.spectator_mode_input[:] = spectator_mode_set.index(self.spectator_mode)
        for s in self.slot_params.keys():
            self.to_predict[s] = np.zeros(self.shape, dtype=np.uint8)

            self.names[s] = Counter()

    def figure_slot_params(self, film_format):
        left_params = BOX_PARAMETERS[film_format]['LEFT_NAME']
        right_params = BOX_PARAMETERS[film_format]['RIGHT_NAME']
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
        print(self.slot_params)

    def process_frame(self, frame, time_point):
        #if time_point % self.time_step != 0:
        #    return
        #cv2.imshow('frame', frame)
        for i,(s, params) in enumerate(self.slot_params.items()):

            x = params['x']
            y = params['y']
            box = frame[y: y + self.params['HEIGHT'],
                  x: x + self.params['WIDTH']]
            box = np.pad(box,((int(self.params['HEIGHT']/2), int(self.params['HEIGHT']/2)),(0,0),(0,0)), mode='constant', constant_values=0)
            if self.debug:
                cv2.imshow('frame_{}'.format(s), box)

            #cv2.imshow('bw_{}'.format(s), bw)
            box = np.transpose(box, axes=(2, 0, 1))
            self.to_predict[s][self.process_index, ...] = box[None]
        if self.debug:
            cv2.waitKey()
        self.process_index += 1

        if self.process_index == self.batch_size:
            self.annotate()
            for s in self.slot_params.keys():
                self.to_predict[s] = np.zeros(self.shape, dtype=np.uint8)
            self.process_index = 0

    def annotate(self):
        if self.process_index == 0:
            return
        for s in self.slot_params.keys():
            loadData(self.images, torch.from_numpy(self.to_predict[s][:self.process_index, ...]).float())
            batch_size = self.images.size(0)
            with torch.no_grad():
                preds = self.model(self.images, self.spectator_mode_input[:self.process_index])

            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds = preds.to('cpu')

            sim_preds = self.model.converter.decode(preds.data, preds_size.data, raw=False)

            names = [x.replace(',', '') for x in sim_preds]
            self.names[s].update(names)

    def generate_names(self):
        output_names = {'left':{}, 'right':{}}
        for s, v in self.names.items():
            print(s)
            print(v)
            output_names[s[0]][s[1]] = max(v, key=lambda x: v[x])
        output_names['left'][0] = 'glister'
        output_names['left'][1] = 'stalk3r'
        output_names['left'][2] = 'oberon'
        output_names['left'][3] = 'woohyal'
        output_names['left'][4] = 'creative'
        output_names['left'][5] = 'bliss'
        output_names['right'][0] = 'kami'
        output_names['right'][1] = 'mer1t'
        output_names['right'][2] = 'jmac'
        output_names['right'][3] = 'sven'
        output_names['right'][4] = 'molly'
        output_names['right'][5] = 'lensa'
        return output_names
