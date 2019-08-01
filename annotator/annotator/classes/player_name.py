import os
import torch
import numpy as np
import cv2
from annotator.annotator.classes.base import BaseAnnotator
from torch.autograd import Variable

from annotator.models.crnn import PlayerNameCRNN
from annotator.training.helper import load_set
from annotator.training.ctc_helper import loadData
from collections import defaultdict, Counter
from annotator.config import sides, BOX_PARAMETERS


class PlayerNameAnnotator(BaseAnnotator):
    resize_factor = 1
    time_step = 5
    batch_size = 200
    identifier = 'player_name'
    box_settings = 'LEFT_NAME'

    def __init__(self, film_format, model_directory, device):
        super(PlayerNameAnnotator, self).__init__(film_format, device)
        self.figure_slot_params(film_format)
        self.model_directory = model_directory
        label_set = load_set(os.path.join(model_directory, 'labels_set.txt'))

        self.model = PlayerNameCRNN(label_set)
        self.model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pth')))
        self.model.eval()
        self.model.to(device)
        self.to_predict = {}
        self.names = {}
        self.shape = (self.batch_size, 3, int(self.params['HEIGHT'] * self.resize_factor * 2),
                      int(self.params['WIDTH'] * self.resize_factor))
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
        for s, params in self.slot_params.items():

            x = params['x']
            y = params['y']
            box = frame[y: y + self.params['HEIGHT'],
                  x: x + self.params['WIDTH']]
            box = np.pad(box,((int(self.params['HEIGHT']/2), int(self.params['HEIGHT']/2)),(0,0),(0,0)), mode='constant', constant_values=0)
            #cv2.imshow('frame_{}'.format(s), box)

            #cv2.imshow('bw_{}'.format(s), bw)
            box = np.transpose(box, axes=(2, 0, 1))
            self.to_predict[s][self.process_index, ...] = box[None]
        #cv2.waitKey()
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
            image = torch.from_numpy(self.to_predict[s]).float().to(self.device)
            batch_size = image.size(0)
            preds = self.model(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.model.converter.decode(preds.data, preds_size.data, raw=False)

            names = [x.replace(',', '') for x in sim_preds]
            self.names[s].update(names)
        print(self.generate_names())

    def generate_names(self):
        output_names = {}
        for s, v in self.names.items():
            print(s)
            print(v)
            output_names[s] = max(v, key=lambda x: v[x])
        return output_names
