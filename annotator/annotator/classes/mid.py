import os
import torch
from annotator.annotator.classes.base import BaseAnnotator, filter_statuses
from annotator.models.cnn import MidCNN
from annotator.training.helper import load_set
from collections import defaultdict


class MidAnnotator(BaseAnnotator):
    resize_factor = 0.5
    batch_size = 200
    time_step = 0.1
    identifier = 'mid'
    box_settings = 'MID'

    def __init__(self, film_format, model_directory, device, debug=False):
        super(MidAnnotator, self).__init__(film_format, device, debug)
        set_paths = {
            'overtime': os.path.join(model_directory, 'overtime_set.txt'),
            'point_status': os.path.join(model_directory, 'point_status_set.txt'),
            'attacking_side': os.path.join(model_directory, 'attacking_side_set.txt'),
            'map': os.path.join(model_directory, 'map_set.txt'),
            'map_mode': os.path.join(model_directory, 'map_mode_set.txt'),
            'round_number': os.path.join(model_directory, 'round_number_set.txt'),
            'spectator_mode': os.path.join(model_directory, 'spectator_mode_set.txt'),
        }

        sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)
        self.model = MidCNN(sets)
        self.model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pth')))
        self.model.eval()
        self.model.to(device)

        self.statuses = {k: [] for k in list(self.model.sets.keys())}

    def annotate(self):
        if self.process_index == 0:
            return
        with torch.no_grad():
            predicteds = self.model({'image': torch.from_numpy(self.to_predict[:self.process_index]).float().to(self.device)})
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
        overtimes = filter_statuses (self.statuses['overtime'], 2)
        point_status = filter_statuses(self.statuses['point_status'], 2)
        actual_overtime = []
        for o in overtimes:
            if o['status'] == 'overtime':
                actual_overtime.append({'begin': o['begin'], 'end': o['end']})
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

        for k in ['attacking_side', 'map', 'map_mode', 'round_number']:
            counts = defaultdict(float)
            for r in self.statuses[k]:
                counts[r['status']] += r['end'] - r['begin']
            print(k, counts)
            out_props[k] = max(counts, key=lambda x: counts[x])
        if out_props['map_mode'] == 'control':
            out_props['attacking_side'] = 'N'
        elif out_props['attacking_side'] == 'left':
            out_props['attacking_side'] = 'L'
        else:
            out_props['attacking_side'] = 'R'

        point_totals = []
        point_flips = []
        if out_props['map_mode'] == 'control':
            for p in point_status:
                if p['status'].startswith('Control'):
                    if p['status'].endswith('neither'):
                        continue
                    if p['status'].endswith('left'):
                        point_flips.append({'time_point': p['begin'], 'controlling_side': 'L'})
                    elif p['status'].endswith('right'):
                        point_flips.append({'time_point': p['begin'], 'controlling_side': 'R'})
        else:
            for p in point_status:
                print(p)
                map_mode = out_props['map_mode'].title()
                if map_mode == 'Assault':
                    points_per_round = 2
                else:
                    points_per_round = 3
                if p['status'].startswith(map_mode):
                    total = int(p['status'].replace(map_mode + '_', '')) - 1
                    print(total)
                    if total:
                        if 'round_number' in round_object:
                            previous_points = points_per_round * int((int(round_object['round_number']) - 1) / 2)
                        else:
                            previous_points = points_per_round * int((int(out_props['round_number']) - 1) / 2)
                        total += previous_points
                        print(total)
                        print(point_totals)
                        if not point_totals or total > point_totals[-1]['point_total']:
                            point_totals.append({'time_point': p['begin'], 'point_total': total})
        out_props['point_flips'] = point_flips
        out_props['point_gains'] = point_totals

        out_props['overtimes'] = actual_overtime
        return out_props
