import os
import torch
from annotator.annotator.classes.base import BaseAnnotator
from annotator.models.cnn import PauseCNN
from annotator.training.helper import load_set
from annotator.annotator.classes.hmm import HMM
import h5py
import numpy as np


class PauseAnnotator(BaseAnnotator):
    time_step = 1
    resize_factor = 0.5
    identifier = 'pause'
    box_settings = 'PAUSE'
    cnn = PauseCNN

    def __init__(self, film_format, model_directory, device):
        super(PauseAnnotator, self).__init__(film_format, device)
        self.model_directory = model_directory
        set_paths = {self.identifier: os.path.join(model_directory, self.identifier+'_set.txt')}
        sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)
        self.model = self.cnn(sets)
        self.model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pth')))
        self.model.eval()
        self.model.to(device)
        prob_path = os.path.join(model_directory, 'hmm_probs.h5')
        self.hmm = HMM(len(sets[self.identifier]))
        with h5py.File(prob_path, 'r') as hf5:
            self.hmm.startprob_ = hf5['{}_init'.format(self.identifier)][:].astype(np.float_)
            trans = hf5['{}_trans'.format(self.identifier)][:].astype(np.float_)
            for i in range(trans.shape[0]):
                if trans[i, i] == 0:
                    trans[i, i] = 1
            self.hmm.transmat_ = trans
        self.probs = []

        self.status = []

    def annotate(self):
        if self.process_index == 0:
            return
        predicteds = self.model({'image': torch.from_numpy(self.to_predict).float().to(self.device)})
        for k, v in predicteds.items():
            v = v.to('cpu').detach()
            self.probs.extend(v.numpy())
            _, predicteds[k] = torch.max(v, 1)
            for t_ind in range(self.batch_size):
                current_time = self.begin_time + (t_ind * self.time_step)
                label = self.model.sets[k][predicteds[k][t_ind]]
                if len(self.status) == 0:
                    self.status.append({'begin': 0, 'end': 0, 'status': label})
                else:
                    if label == self.status[-1]['status']:
                        self.status[-1]['end'] = current_time
                    else:
                        self.status.append(
                            {'begin': current_time, 'end': current_time, 'status': label})

    def generate_final_statuses(self):
        self.status = []
        p = np.array(self.probs).astype(np.float_)
        log, z = self.hmm.decode(p)
        for i, z1 in enumerate(z):
            current_time = i * self.time_step
            label = self.model.sets[self.identifier][z1]
            if len(self.status) == 0:
                self.status.append({'begin': 0, 'end': 0, 'status': label})
            else:
                if label == self.status[-1]['status']:
                    self.status[-1]['end'] = current_time
                else:
                    self.status.append(
                        {'begin': current_time, 'end': current_time, 'status': label})
        new_status = []
        threshold = 5
        for i, x in enumerate(self.status):
            if not new_status:
                if i < len(self.status) - 1 and \
                        x['end'] - x['begin'] < self.status[i+1]['end'] - self.status[i+1]['begin'] and i == 0:
                    continue
                else:
                    new_status.append(x)
            else:
                if x['status'] == new_status[-1]['status']:
                    new_status[-1]['end'] = x['end']
                elif x['end'] - x['begin'] > threshold and x['status'] != 'n/a':
                    new_status.append(x)
        new_status[0]['begin'] = 0
        return new_status

    def generate_pauses(self):
        status = self.generate_final_statuses()
        print('pause', status)
        events = []
        for interval in status:
            if interval['status'] == self.identifier:
                events.append(interval)
        return events
