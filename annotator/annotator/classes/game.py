import os
import torch
import numpy as np
from annotator.annotator.classes.base import BaseAnnotator
from annotator.models.cnn import GameCNN
from annotator.training.helper import load_set
from annotator.training.ctc_helper import loadData
from torch.autograd import Variable
from annotator.annotator.classes.base import filter_statuses


class InGameAnnotator(BaseAnnotator):
    time_step = 0.5
    batch_size = 100
    resize_factor = 0.2
    identifier = 'game'
    box_settings = 'WINDOW'

    def __init__(self, film_format, model_directory, device):
        super(InGameAnnotator, self).__init__(film_format, device)
        self.model_directory = model_directory
        set_paths = {'game': os.path.join(model_directory, 'game_set.txt'),
            'map': os.path.join(model_directory, 'map_set.txt'),
            #'film_format': os.path.join(model_directory, 'film_format_set.txt'),
            'left_color': os.path.join(model_directory, 'left_color_set.txt'),
            'right_color': os.path.join(model_directory, 'right_color_set.txt'),
            'spectator_mode': os.path.join(model_directory, 'spectator_mode_set.txt'),
            'left': os.path.join(model_directory, 'left_set.txt'),
            'right': os.path.join(model_directory, 'right_set.txt'),
                     }
        torch.cuda.empty_cache()
        sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)
        self.model = GameCNN(sets)
        self.model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pth')))
        self.model.eval()
        self.model.to(device)
        self.images = Variable(torch.FloatTensor(self.batch_size, 3, self.image_height, self.image_width).to(device))
        #self.soft_maxes = {k: [] for k in sets.keys()}
        self.status = {k:[] for k in sets.keys()}

    def annotate(self):
        if self.process_index == 0:
            return
        print(self.to_predict.shape)
        loadData(self.images, torch.from_numpy(self.to_predict).float())
        with torch.no_grad():
            predicteds = self.model({'image': self.images})
        print(self.begin_time)
        for k, v in predicteds.items():
            #if k == 'map':
            #    self.soft_maxes[k].extend(v.cpu().numpy())
            _, predicteds[k] = torch.max(v.cpu(), 1)
            for t_ind in range(self.batch_size):
                current_time = self.begin_time + (t_ind * self.time_step)
                label = self.model.sets[k][predicteds[k][t_ind]]
                if len(self.status[k]) == 0:
                    self.status[k].append({'begin': 0, 'end': 0, 'status': label})
                else:
                    if label == self.status[k][-1]['status']:
                        self.status[k][-1]['end'] = current_time
                    else:
                        self.status[k].append(
                            {'begin': current_time, 'end': current_time, 'status': label})

    def generate_rounds(self):
        import time
        from collections import Counter
        data = {}
        for k, intervals in self.status.items():
            print(k)
            for interval in intervals:
                begin_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['begin']))
                end_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['end']))
                print('      ', '{}-{}: {}'.format(begin_timestamp, end_timestamp, interval['status']))
        thresholds = {'game': {'not_game': 10, 'game': 7, 'pause':5, 'replay':4, 'smaller_window': 4},
                      'left': 1,
                      'right': 1,
                      }
        new_status = {}
        for k in self.status.keys():
            if k not in thresholds: # Don't filter
                new_status[k] = self.status[k]
            else:
                new_status[k] = filter_statuses(self.status[k], thresholds[k])
        print('FILTERED')
        for k, intervals in new_status.items():
            print(k)
            for interval in intervals:
                begin_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['begin']))
                end_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['end']))
                print('      ', '{}-{}: {}'.format(begin_timestamp, end_timestamp, interval['status']))
        cur_round = None
        rounds = []
        for interval in new_status['game']:
            if cur_round is None and interval['status'] == 'game':
                cur_round = {'begin': interval['begin'], 'end': interval['end'], 'pauses':[], 'replays':[],
                             'smaller_windows':[], 'left_zooms':[], 'right_zooms':[]}
            elif cur_round is not None and interval['status'] == 'not_game':
                rounds.append(cur_round)
                cur_round = None
            elif cur_round is not None:
                if interval['status'] == 'game':
                    cur_round['end'] = interval['end']
                elif interval['status'] == 'pause':
                    cur_round['pauses'].append({'begin': interval['begin'] - cur_round['begin'],
                                                'end': interval['end'] - cur_round['begin']})
                elif interval['status'] == 'replay':
                    cur_round['replays'].append({'begin': interval['begin'] - cur_round['begin'],
                                                'end': interval['end'] - cur_round['begin']})
                elif interval['status'] == 'smaller_window':
                    cur_round['smaller_windows'].append({'begin': interval['begin'] - cur_round['begin'],
                                                'end': interval['end'] - cur_round['begin']})
        if cur_round is not None:
            rounds.append(cur_round)
        rounds = [r for r in rounds if r['end'] - r['begin'] > 30]
        for i, r in enumerate(rounds):
            for side in ['left', 'right']:
                for interval in new_status[side]:
                    if interval['begin'] < r['begin']:
                        continue
                    if interval['begin'] > r['end']:
                        break
                    if interval['status'] == 'zoom':
                        r[side+'_zooms'].append({'begin': interval['begin'] - r['begin'],
                                                'end': interval['end'] - r['begin']})
            #begin_index = int(r['begin'] / InGameAnnotator.time_step)
            #end_index = int(r['end'] / InGameAnnotator.time_step)
            #print(r['begin'], r['end'])
            #print(begin_index, end_index)
            #a = np.array(self.soft_maxes['map'][begin_index:end_index])
            #print(a.shape)
            #a_mean = np.mean(a, axis=0)
            #print(a_mean.shape)
            #print(a_mean)
            #r['mean_map'] = self.model.sets['map'][np.argmax(a_mean)]
            count_dict = Counter()
            for interval in new_status['map']:
                if interval['begin'] < r['begin']:
                    continue
                if interval['begin'] > r['end']:
                    break
                if interval['status'] != 'n/a':
                    count_dict[interval['status']] += interval['end'] - interval['begin']
            print(r['begin'], r['end'])
            print(time.strftime('%H:%M:%S', time.gmtime(r['begin'])))
            print(time.strftime('%H:%M:%S', time.gmtime(r['end'])))
            print(count_dict)
            if not count_dict:
                if i > 0:
                    r['map'] = rounds[i-1]['map']
            else:
                r['map'] = max(count_dict, key=lambda x: count_dict[x])

        count_dict = Counter()
        for interval in new_status['spectator_mode']:
            if interval['status'] != 'n/a':
                count_dict[interval['status']] += interval['end'] - interval['begin']
        data['spectator_mode'] = max(count_dict, key=lambda x: count_dict[x])
        count_dict = Counter()
        #for interval in new_status['film_format']:
        #    if interval['status'] != 'n/a':
        #        count_dict[interval['status']] += interval['end'] - interval['begin']
        #data['film_format'] = max(count_dict, key=lambda x: count_dict[x])
        count_dict = Counter()
        for interval in new_status['left_color']:
            if interval['status'] != 'n/a':
                count_dict[interval['status']] += interval['end'] - interval['begin']
        data['left_color'] = max(count_dict, key=lambda x: count_dict[x])
        count_dict = Counter()
        for interval in new_status['right_color']:
            if interval['status'] != 'n/a':
                count_dict[interval['status']] += interval['end'] - interval['begin']
        data['right_color'] = max(count_dict, key=lambda x: count_dict[x])
        return data, rounds
