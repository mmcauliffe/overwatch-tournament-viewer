import os
import torch
import numpy as np
from annotator.training.helper import load_set
from annotator.models.rnn import StatusGRU
from annotator.annotator.classes.player_status import PlayerStatusAnnotator
from torch.autograd import Variable
from annotator.training.ctc_helper import loadData


class PlayerRNNAnnotator(PlayerStatusAnnotator):
    time_step = 0.1
    batch_size = 100

    def __init__(self, film_format, model_directory, device, left_color, right_color, player_names, spectator_mode='O'):
        super(PlayerStatusAnnotator, self).__init__(film_format, device)
        self.figure_slot_params(film_format)
        self.model_directory = model_directory
        self.left_team_color = left_color
        self.right_team_color = right_color
        self.player_names = player_names
        self.spectator_mode = spectator_mode
        set_paths = {
            'hero': os.path.join(model_directory, 'hero_set.txt'),
            'alive': os.path.join(model_directory, 'alive_set.txt'),
            'ult': os.path.join(model_directory, 'ult_set.txt'),
            'status': os.path.join(model_directory, 'status_set.txt'),
            'antiheal': os.path.join(model_directory, 'antiheal_set.txt'),
            'immortal': os.path.join(model_directory, 'immortal_set.txt'),

        }
        sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)
        input_set_files = {
            'color': os.path.join(model_directory, 'color_set.txt'),
            'spectator_mode': os.path.join(model_directory, 'spectator_mode_set.txt'),
             }
        input_sets = {}
        for k, v in input_set_files.items():
            input_sets[k] = load_set(v)
        self.model = StatusGRU(sets, input_sets)
        self.model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pth')))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)

        self.statuses = {}
        self.shape = (12, self.batch_size, 3, int(self.params['HEIGHT'] * self.resize_factor),
                      int(self.params['WIDTH'] * self.resize_factor))
        self.images = Variable(torch.FloatTensor(12,self.batch_size, 3, int(self.params['HEIGHT'] * self.resize_factor),
                      int(self.params['WIDTH'] * self.resize_factor)).to(device))
        self.inputs = {}
        self.inputs['spectator_mode'] = Variable(torch.LongTensor(12,self.batch_size).to(device))
        self.inputs['spectator_mode'][:, :] = input_sets['spectator_mode'].index(self.spectator_mode)
        self.inputs['color'] = Variable(torch.LongTensor(12, self.batch_size).to(device))
        self.inputs['color'][:6,:] = input_sets['color'].index(self.left_team_color)
        self.inputs['color'][6:,:] = input_sets['color'].index(self.right_team_color)

        self.to_predict = np.zeros(self.shape, dtype=np.uint8)
        for s in self.slot_params.keys():
            self.statuses[s] = {k: [] for k in list(self.model.sets.keys())}


    def process_frame(self, frame):
        #cv2.imshow('frame', frame)
        for i, (s, params) in enumerate(self.slot_params.items()):

            x = params['x']
            y = params['y']
            box = frame[y: y + self.params['HEIGHT'],
                  x: x + self.params['WIDTH']]
            #cv2.imshow('frame_{}'.format(s), box)

            box = np.transpose(box, axes=(2, 0, 1))
            self.to_predict[i,self.process_index, ...] = box[None]
        #cv2.waitKey()
        self.process_index += 1

        if self.process_index == self.batch_size:
            self.annotate()
            self.reset(self.begin_time + (self.batch_size * self.time_step))

    def annotate(self):
        import time
        begin = time.time()
        if self.process_index == 0:
            return
        #print(s)
        b = time.time()
        loadData(self.images, torch.from_numpy(self.to_predict).float())
        ins = {'image': self.images, 'spectator_mode': self.inputs['spectator_mode'],
               'color':self.inputs['color']}

        predicteds = self.model(ins)
        #print('got predictions:', time.time()-b)
        b = time.time()
        for k, v in predicteds.items():
            #print(k)
            #print(predicteds[k])
            _, predicteds[k] = torch.max(v.to('cpu'), 2)
            for si, s in enumerate(self.slot_params.keys()):
                for t_ind in range(self.process_index - 1):
                    #cv2.imshow('frame_{}'.format(t_ind), np.transpose(self.to_predict[s][t_ind], axes=(1, 2, 0)))
                    current_time = self.begin_time + (t_ind * self.time_step)
                    #print(current_time)
                    label = self.model.sets[k][predicteds[k][si, t_ind]]
                    #print(label)
                    if len(self.statuses[s][k]) == 0:
                        self.statuses[s][k].append({'begin': 0, 'end': 0, 'status': label})
                    else:
                        if label == self.statuses[s][k][-1]['status']:
                            self.statuses[s][k][-1]['end'] = current_time
                        else:
                            self.statuses[s][k].append(
                                {'begin': current_time, 'end': current_time, 'status': label})
            #cv2.waitKey()
        #print('created statuses:', time.time()-b)
        print('Status annotate took: ', time.time() - begin)