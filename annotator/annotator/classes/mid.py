import os
import torch
from annotator.annotator.classes.base import BaseAnnotator, filter_statuses
from annotator.models.cnn import MidCNN
from annotator.training.helper import load_set
from collections import defaultdict
from torch.autograd import Variable
from annotator.training.ctc_helper import loadData


class MidAnnotator(BaseAnnotator):
    resize_factor = 0.5
    batch_size = 200
    time_step = 0.1
    identifier = 'mid'
    box_settings = 'MID'

    def __init__(self, film_format, model_directory, device, spectator_mode, map, attacking_side, debug=False):
        super(MidAnnotator, self).__init__(film_format, device, debug)
        print('=== SETTING UP MID ANNOTATOR ===')
        self.spectator_mode = spectator_mode
        self.map = map
        self.attacking_side = attacking_side
        set_paths = {
            'overtime': os.path.join(model_directory, 'overtime_set.txt'),
            'point_status': os.path.join(model_directory, 'point_status_set.txt'),
        }
        input_set_paths = {
            #'spectator_mode': os.path.join(model_directory, 'spectator_mode_set.txt'),
            'map': os.path.join(model_directory, 'map_set.txt'),
            'attacking_side': os.path.join(model_directory, 'attacking_side_set.txt'),
        }

        sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)

        input_sets = {}
        for k, v in input_set_paths.items():
            input_sets[k] = load_set(v)

        self.model = MidCNN(sets, input_sets)
        spec_dir = os.path.join(model_directory, self.spectator_mode)
        if os.path.exists(spec_dir):
            model_path = os.path.join(spec_dir, 'model.pth')
            print('Using {} mid model!'.format(self.spectator_mode))
        else:
            model_path = os.path.join(model_directory, 'model.pth')
            print('Using base mid model!')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.image_height = int(self.params['HEIGHT'] * self.resize_factor)
        self.image_width = int(self.params['WIDTH'] * self.resize_factor)
        self.model.to(device)
        self.images = Variable(
            torch.FloatTensor(self.batch_size, 3, self.image_height, self.image_width).to(device))
        self.inputs = {}

        self.inputs['map'] = Variable(torch.LongTensor(self.batch_size).to(device))
        self.inputs['map'][:] = input_sets['map'].index(self.map)

        self.inputs['attacking_side'] = Variable(torch.LongTensor(self.batch_size).to(device))
        self.inputs['attacking_side'][:] = input_sets['attacking_side'].index(self.attacking_side)

        self.statuses = {k: [] for k in list(self.model.sets.keys())}

    def annotate(self):
        if self.process_index == 0:
            return
        with torch.no_grad():

            t = torch.from_numpy(self.to_predict).float()
            t = ((t / 255) - 0.5) / 0.5
            loadData(self.images, t)

            ins = {'image': self.images,
                   'map': self.inputs['map'], 'attacking_side': self.inputs['attacking_side']}
            predicteds = self.model(ins)
        for k, v in predicteds.items():
            _, predicteds[k] = torch.max(v.to('cpu'), 1)
            for t_ind in range(self.process_index):
                current_time = self.begin_time + (t_ind * self.time_step)
                label = self.model.sets[k][predicteds[k][t_ind]]
                if len(self.statuses[k]) == 0:
                    self.statuses[k].append({'begin': 0, 'end': 0, 'status': label})
                else:
                    if label == self.statuses[k][-1]['status']:
                        self.statuses[k][-1]['end'] = current_time
                    else:
                        self.statuses[k].append(
                            {'begin': current_time, 'end': current_time, 'status': label})

    def generate_round_properties(self, round_object):
        overtimes = filter_statuses(self.statuses['overtime'], 2)
        point_status = filter_statuses(self.statuses['point_status'], 2)
        actual_overtime = []
        for o in overtimes:
            if o['status'] == 'overtime':
                actual_overtime.append({'begin': o['begin'], 'end': o['end'] + 0.1})
        out_props = {}
        actual_points = []
        for i, r in enumerate(self.statuses['point_status']):
            if r['end'] - r['begin'] < 2:
                continue
            if len(actual_points) and r['status'] == actual_points[-1]['status']:
                actual_points[-1]['end'] = r['end']
            else:
                if len(actual_points) and actual_points[-1]['end'] != r['begin']:
                    actual_points[-1]['end'] = r['begin']
                actual_points.append(r)

        point_totals = []
        point_flips = []
        for p in point_status:
            if p['status'].startswith('Control'):
                if p['status'].endswith('neither'):
                    continue
                if p['status'].endswith('left'):
                    point_flips.append({'time_point': p['begin'], 'controlling_side': 'L'})
                elif p['status'].endswith('right'):
                    point_flips.append({'time_point': p['begin'], 'controlling_side': 'R'})
            else:
                map_mode, total = p['status'].split('_')
                total = int(total) - 1
                if map_mode == 'Assault':
                    points_per_round = 2
                else:
                    points_per_round = 3
                if total:
                    if 'round_number' in round_object:
                        previous_points = points_per_round * int((round_object['round_number'] - 1) / 2)
                    else:
                        previous_points = points_per_round * int((round_object['round_number'] - 1) / 2)
                    total += previous_points
                    print(total)
                    print(point_totals)
                    if not point_totals or total > point_totals[-1]['point_total']:
                        point_totals.append({'time_point': p['begin'], 'point_total': total})
        out_props['point_flips'] = point_flips
        out_props['point_gains'] = point_totals

        out_props['overtimes'] = actual_overtime
        return out_props
