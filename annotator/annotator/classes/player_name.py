import os
import torch
import numpy as np
import cv2
import jamotools
from annotator.annotator.classes.base import BaseAnnotator
from torch.autograd import Variable

from annotator.models.crnn import PlayerNameCRNN, KillFeedCRNN
from annotator.training.helper import load_set
from annotator.training.ctc_helper import loadData
from collections import defaultdict, Counter
from annotator.config import sides, BOX_PARAMETERS


class PlayerNameAnnotator(BaseAnnotator):
    resize_factor = 2
    time_step = 0.2
    batch_size = 100
    identifier = 'player_name'
    box_settings = 'LEFT_NAME'

    def __init__(self, film_format, model_directory, device, spectator_mode, debug=False):
        super(PlayerNameAnnotator, self).__init__(film_format, device, debug)
        self.spectator_mode = spectator_mode
        self.figure_slot_params(film_format)
        self.model_directory = model_directory
        label_set = load_set(os.path.join(model_directory, 'labels_set.txt'))
        self.model = PlayerNameCRNN(label_set)
        self.model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pth')))
        self.model.eval()
        self.model.to(device)
        self.to_predict = {}
        self.names = {}
        self.image_height = self.params['HEIGHT'] * self.resize_factor
        self.image_width = self.params['WIDTH'] * self.resize_factor
        self.shape = (self.batch_size, 3, self.image_height, self.image_width)
        self.images = Variable(torch.FloatTensor(self.batch_size, 3, self.image_height, self.image_width).to(device))
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
            box = cv2.resize(box, (0, 0),fx=self.resize_factor, fy=self.resize_factor)
            #box = np.pad(box,((int(self.params['HEIGHT']/2), int(self.params['HEIGHT']/2)),(0,0),(0,0)), mode='constant', constant_values=0)
            if self.debug:
                cv2.imshow('frame_{}'.format(s), box)

            #cv2.imshow('bw_{}'.format(s), bw)
            box = np.transpose(box, axes=(2, 0, 1))
            self.to_predict[s][self.process_index, ...] = box[None]
        #if self.debug:
        #    cv2.waitKey()
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
            t = torch.from_numpy(self.to_predict[s][:self.process_index, ...]).float()
            t = ((t / 255) - 0.5) / 0.5
            loadData(self.images, t)
            batch_size = self.images.size(0)
            with torch.no_grad():
                preds = self.model(self.images)

            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds = preds.to('cpu')

            sim_preds = self.model.converter.decode(preds.data, preds_size.data, raw=False)

            names = [x.replace(',', '') for x in sim_preds]
            self.names[s].update(names)

    def generate_names(self, teams):
        fixes = {
            'hooeg': 'hooreg',
            'jecsee': 'jecse',
            'nosmie': 'nosmite',
            'dacdo': 'daco',
            'gegturi': 'geguri',
            'gegture': 'geguri',
            'coneipten': 'onlywish',
            'conewphek': 'onlywish',
            'onwoish': 'onlywish',
            'onfoish': 'onlywish',
            'onoish': 'onlywish',
            'fradi': 'fragi',
            'fiollt': 'highly',
            'fioplt': 'highly',
            'piopfet': 'highly',
            'fiolhply': 'highly',
            'fioply': 'highly',
            'slaler': 'scaler',
                 }
        output_names = {'left': {}, 'right': {}}
        counts = {'left': [0, 0], 'right': [0, 0]}
        team_one_names = [jamotools.split_syllables(x['name']) for x in teams[0]['players']]
        team_two_names = [jamotools.split_syllables(x['name']) for x in teams[1]['players']]
        print(team_one_names)
        print(team_two_names)
        team_one_taken = []
        team_two_taken = []
        for s, v in self.names.items():
            print(s)
            print(v)
            for n, c in v.items():
                if n in team_one_names + team_one_names:
                    name = n
                    break
            else:
                name = max(v, key=lambda x: v[x])
            if name in fixes:
                name = fixes[name]
            if name in team_one_names:
                counts[s[0]][0] += 1
                team_one_taken.append(name)
            elif name in team_two_names:
                counts[s[0]][1] += 1
                team_two_taken.append(name)
            output_names[s[0]][s[1]] = name
        import editdistance
        team_one_left = counts['left'][0] > counts['left'][1]
        print(team_one_left)
        print(counts)
        for side, side_dict in output_names.items():
            if team_one_left and side == 'left':
                team_names = [x for x in team_one_names if x not in team_one_taken]
                taken = team_one_taken
            elif team_one_left:
                team_names = [x for x in team_two_names if x not in team_two_taken]
                taken = team_two_taken
            elif not team_one_left and side == 'left':
                team_names = [x for x in team_two_names if x not in team_two_taken]
                taken = team_two_taken
            else:
                team_names = [x for x in team_one_names if x not in team_one_taken]
                taken = team_one_taken
            for i, name in side_dict.items():
                if name in taken:
                    continue
                best_score = 10000
                best_name = name
                for t_name in team_names:
                    dist = editdistance.eval(name, t_name)
                    if dist < best_score:
                        best_name = t_name
                        best_score = dist
                output_names[side][i] = best_name
                taken.append(best_name)

        for side, side_dict in output_names.items():
            for i, name in side_dict.items():
                output_names[side][i] = jamotools.join_jamos(name)
        print(output_names)
        print(team_one_taken)
        print(team_two_taken)
        return output_names
