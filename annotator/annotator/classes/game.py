import os
import torch
from annotator.annotator.classes.base import BaseAnnotator
from annotator.models.cnn import MidCNN
from annotator.training.helper import load_set


class InGameAnnotator(BaseAnnotator):
    time_step = 1
    resize_factor = 0.5
    identifier = 'game'
    box_settings = 'MID'

    def __init__(self, film_format, model_directory, device):
        super(InGameAnnotator, self).__init__(film_format, device)
        self.model_directory = model_directory
        set_paths = {'game': os.path.join(model_directory, 'game_set.txt')}
        sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)
        self.model = MidCNN(sets)
        self.model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pth')))
        self.model.eval()
        self.model.to(device)

        self.status = []

    def annotate(self):
        if self.process_index == 0:
            return
        predicteds = self.model({'image': torch.from_numpy(self.to_predict).float().to(self.device)})
        for k, v in predicteds.items():
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
