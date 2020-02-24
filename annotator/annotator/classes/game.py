import os
import torch
import numpy as np
from annotator.annotator.classes.base import BaseAnnotator
from annotator.models.cnn import GameCNN
from annotator.training.helper import load_set
from annotator.training.ctc_helper import loadData
from torch.autograd import Variable
from annotator.annotator.classes.base import filter_statuses, coalesce_statuses


class InGameAnnotator(BaseAnnotator):
    time_step = 0.5
    batch_size = 25
    resize_factor = 0.2
    identifier = 'game'
    box_settings = 'WINDOW'

    def __init__(self, film_format, spectator_mode, model_directory, device, use_spec_modes=True):
        super(InGameAnnotator, self).__init__(film_format, device)
        self.model_directory = model_directory
        self.spectator_mode = spectator_mode
        set_paths = {
            'game': os.path.join(model_directory, 'game_set.txt'),
            'map': os.path.join(model_directory, 'map_set.txt'),
            'submap': os.path.join(model_directory, 'submap_set.txt'),
            'left_color': os.path.join(model_directory, 'left_color_set.txt'),
            'right_color': os.path.join(model_directory, 'right_color_set.txt'),
            'left': os.path.join(model_directory, 'left_set.txt'),
            'right': os.path.join(model_directory, 'right_set.txt'),
            'attacking_side': os.path.join(model_directory, 'attacking_side_set.txt'),
            #'film_format': os.path.join(model_directory, 'film_format_set.txt'),
            #'spectator_mode': os.path.join(model_directory, 'spectator_mode_set.txt'),
        }
        input_set_paths = {
            #'film_format': os.path.join(model_directory, 'film_format_set.txt'),
            #'spectator_mode': os.path.join(model_directory, 'spectator_mode_set.txt'),
        }
        torch.cuda.empty_cache()
        sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)
        input_sets = {}
        for k, v in input_set_paths.items():
            input_sets[k] = load_set(v)
        self.model = GameCNN(sets, input_sets)
        spec_dir = os.path.join(model_directory, self.spectator_mode)
        if os.path.exists(spec_dir) and use_spec_modes:
            model_path = os.path.join(spec_dir, 'model.pth')
            print('Using {} game model!'.format(self.spectator_mode))
        else:
            model_path = os.path.join(model_directory, 'model.pth')
            print('Using base game model!')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(device)
        self.images = Variable(torch.FloatTensor(self.batch_size, 3, self.image_height, self.image_width).to(device))
        self.inputs = {}
        #self.inputs['spectator_mode'] = torch.LongTensor(self.batch_size).to(device)
        #self.inputs['spectator_mode'][:, :] = input_sets['spectator_mode'].index(self.spectator_mode)
        #self.inputs['film_format'] = torch.LongTensor(self.batch_size).to(device)
        #self.inputs['film_format'][:, :] = input_sets['film_format'].index(self.spectator_mode)

        # self.soft_maxes = {k: [] for k in sets.keys()}
        self.status = {k: [] for k in sets.keys()}

    def reset_status(self):
        self.status = {k: [] for k in self.model.sets.keys()}

    def annotate(self):
        if self.process_index == 0:
            return
        t = torch.from_numpy(self.to_predict[:self.process_index, ...]).float()
        t = ((t / 255) - 0.5) / 0.5
        loadData(self.images, t)
        game_inds = []
        with torch.no_grad():
            predicteds = self.model({'image': self.images,
                                     #'spectator_mode': self.inputs['spectator_mode'],
                                     #'film_format': self.inputs['film_format'],
                                     })
            for k, v in predicteds.items():
                _, predicteds[k] = torch.max(v.cpu(), 1)
                for t_ind in range(predicteds[k].size(0)):
                    current_time = self.begin_time + (t_ind * self.time_step)
                    label = self.model.sets[k][predicteds[k][t_ind]]
                    if len(self.status[k]) == 0:
                        self.status[k].append({'begin': self.begin_time, 'end': self.begin_time, 'status': label})
                    else:
                        if label == self.status[k][-1]['status']:
                            self.status[k][-1]['end'] = current_time
                        else:
                            self.status[k].append(
                                {'begin': current_time, 'end': current_time, 'status': label})

    def get_earliest(self, set_name, value):
        for interval in self.status[set_name]:
            if interval['status'] == value:
                return interval['begin']
        return None

    def get_latest(self, set_name, value):
        for interval in sorted(self.status[set_name], key=lambda x: x['begin'], reverse=True):
            if interval['status'] == value:
                return interval['end']
        return None

    def generate_rounds(self, detect_short_pauses=True):
        import time
        from collections import Counter
        data = {}
        self.status['game'] = filter_statuses(self.status['game'], {'not_game': 1})
        for k, intervals in self.status.items():
            print(k)
            for interval in intervals:
                begin_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['begin']))
                end_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['end']))
                print('      ', '{}-{}: {}'.format(begin_timestamp, end_timestamp, interval['status']))
        thresholds = {'game': {'not_game': 10, 'game': 30, 'pause': 1, 'replay': 1, 'smaller_window': 4},
                      'left': 1,
                      'right': 1,
                      }
        if not detect_short_pauses:
            self.status['game'] = filter_statuses(self.status['game'], {'not_game': 4})
            print('INITIAL FILTERED', self.status['game'][:10])
        side_mapping = {'left': 'L', 'right': 'R', 'neither': 'N'}
        for i, interval in enumerate(self.status['game']):
            duration = interval['end'] - interval['begin']
            if interval['status'] == 'not_game':
                if duration <= 4:
                    if i != 0 and self.status['game'][i - 1]['status'] == 'pause':
                        self.status['game'][i]['status'] = 'pause'
                    if i < len(self.status['game']) - 1 and self.status['game'][i + 1]['status'] == 'pause':
                        self.status['game'][i]['status'] = 'pause'
                    if duration >= 1 and i != 0 and self.status['game'][i - 1]['status'] == 'game' and i < len(
                            self.status['game']) - 1 and self.status['game'][i + 1]['status'] == 'game':
                        self.status['game'][i]['status'] = 'pause_not_game'
                elif i != 0 and self.status['game'][i - 1]['status'] == 'pause':
                    self.status['game'][i]['status'] = 'pause_not_game'

        self.status['game'] = coalesce_statuses(self.status['game'])
        new_status = {}
        for k in self.status.keys():
            if k not in thresholds:  # Don't filter
                new_status[k] = self.status[k]
            else:
                new_status[k] = filter_statuses(self.status[k], thresholds[k], protect_initial=True)
        print('FILTERED')
        for k, intervals in new_status.items():
            print(k)
            for interval in intervals:
                begin_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['begin']))
                end_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['end']))
                print('      ', '{}-{}: {}'.format(begin_timestamp, end_timestamp, interval['status']))
        cur_round = None
        rounds = []
        currently_paused = False
        for i, interval in enumerate(new_status['game']):
            if cur_round is None and interval['status'] in ['game', 'smaller_window', 'pause_split screen']:
                beg = interval['begin']
                sm = []
                pauses = []
                if interval['status'] == 'smaller_window':
                    if new_status['game'][i - 1]['status'] == 'not_game':
                        beg = new_status['game'][i - 1]['end']
                    sm.append({'begin': interval['begin'] - beg, 'end': interval['end'] - beg})
                if interval['status'] == 'pause_split screen':
                    if new_status['game'][i - 1]['status'] == 'not_game':
                        beg = new_status['game'][i - 1]['end']
                    pauses.append({'begin': interval['begin'] - beg,
                                   'type': 'Split screen',
                                   'end': interval['end'] - beg})
                cur_round = {'begin': beg, 'end': interval['end'], 'pauses': pauses, 'replays': [],
                             'smaller_windows': sm, 'left_zooms': [], 'right_zooms': []}
            elif cur_round is not None:
                if interval['status'] == 'pause' and not currently_paused:
                    currently_paused = True
                    begin = interval['begin'] - cur_round['begin']
                    end = interval['end'] - cur_round['begin']
                    if interval['status'] == 'pause':
                        cur_round['pauses'].append({'begin': begin,
                                                    'type': 'Ingame pause',
                                                    'end': end})
                elif currently_paused:
                    begin = interval['begin'] - cur_round['begin']
                    end = interval['end'] - cur_round['begin']
                    if interval['status'] == 'game':
                        cur_round['pauses'][-1]['end'] = begin
                        currently_paused = False
                        cur_round['end'] = interval['end']
                    elif interval['status'] == 'pause':
                        cur_round['pauses'][-1]['end'] = begin
                        cur_round['pauses'].append({'begin': begin,
                                                    'type': 'Ingame pause',
                                                    'end': end})
                    else:
                        if cur_round['pauses'][-1]['type'] != 'Out of game pause':
                            cur_round['pauses'][-1]['end'] = begin
                            cur_round['pauses'].append({'begin': begin,
                                                        'type': 'Out of game pause',
                                                        'end': end})
                        else:
                            cur_round['pauses'][-1]['end'] = end
                elif interval['status'] == 'not_game' and not currently_paused:
                    rounds.append(cur_round)
                    cur_round = None
                else:
                    if interval['status'] == 'game':
                        cur_round['end'] = interval['end']
                    elif i < len(new_status['game']) and new_status['game'][i + 1]['status'] == 'game':
                        begin = interval['begin'] - cur_round['begin']
                        end = interval['end'] - cur_round['begin']
                        if interval['status'] == 'pause':
                            cur_round['pauses'].append({'begin': begin,
                                                        'end': end})
                        elif interval['status'] == 'replay':
                            cur_round['replays'].append({'begin': begin,
                                                         'end': end})
                        elif interval['status'] == 'smaller_window':
                            cur_round['smaller_windows'].append({'begin': begin,
                                                                 'end': end})

        if cur_round is not None:
            rounds.append(cur_round)
        rounds = [r for r in rounds if r['end'] - r['begin'] > 20]
        for i, r in enumerate(rounds):
            for side in ['left', 'right']:
                for interval in new_status[side]:
                    if interval['begin'] < r['begin']:
                        continue
                    if interval['begin'] > r['end']:
                        break
                    if interval['status'] == 'zoom':
                        r[side + '_zooms'].append({'begin': interval['begin'] - r['begin'],
                                                   'end': interval['end'] - r['begin']})
            # begin_index = int(r['begin'] / InGameAnnotator.time_step)
            # end_index = int(r['end'] / InGameAnnotator.time_step)
            # print(r['begin'], r['end'])
            # print(begin_index, end_index)
            # a = np.array(self.soft_maxes['map'][begin_index:end_index])
            # print(a.shape)
            # a_mean = np.mean(a, axis=0)
            # print(a_mean.shape)
            # print(a_mean)
            # r['mean_map'] = self.model.sets['map'][np.argmax(a_mean)]
            count_dict = Counter()
            for interval in new_status['submap']:
                if interval['begin'] < r['begin']:
                    continue
                if interval['begin'] > r['end']:
                    break
                if interval['status'] != 'n/a':
                    count_dict[interval['status']] += interval['end'] - interval['begin']
            if not count_dict:
                for interval in new_status['submap']:
                    if interval['begin'] < r['end'] and interval['end'] > r['begin'] and interval['status'] != 'n/a':
                        count_dict[interval['status']] += interval['end'] - interval['begin']
            r['submap'] = max(count_dict, key=lambda x: count_dict[x])
            count_dict = Counter()
            for interval in new_status['map']:
                if interval['begin'] < r['begin']:
                    continue
                if interval['begin'] > r['end']:
                    break
                if interval['status'] != 'n/a':
                    count_dict[interval['status']] += interval['end'] - interval['begin']
            if not count_dict:
                for interval in new_status['map']:
                    if interval['begin'] < r['end'] and interval['end'] > r['begin'] and interval['status'] != 'n/a':
                        count_dict[interval['status']] += interval['end'] - interval['begin']
            print(r['begin'], r['end'])
            print(time.strftime('%H:%M:%S', time.gmtime(r['begin'])))
            print(time.strftime('%H:%M:%S', time.gmtime(r['end'])))
            print(count_dict)
            if not count_dict:
                if i > 0:
                    r['map'] = rounds[i - 1]['map']
            else:
                r['map'] = max(count_dict, key=lambda x: count_dict[x])
            count_dict = Counter()
            for interval in new_status['attacking_side']:
                if interval['begin'] < r['begin']:
                    continue
                if interval['begin'] > r['end']:
                    break
                if interval['status'] != 'n/a':
                    count_dict[interval['status']] += interval['end'] - interval['begin']
            if not count_dict:
                for interval in new_status['attacking_side']:
                    if interval['begin'] < r['end'] and interval['end'] > r['begin'] and interval['status'] != 'n/a':
                        count_dict[interval['status']] += interval['end'] - interval['begin']
            print(new_status['attacking_side'])
            if not count_dict:
                r['attacking_side'] = 'neither'
            else:
                r['attacking_side'] = max(count_dict, key=lambda x: count_dict[x])
            r['attacking_side'] = side_mapping[r['attacking_side']]
        #count_dict = Counter()
        #for interval in new_status['spectator_mode']:
        #    if interval['status'] != 'n/a':
        #        count_dict[interval['status']] += interval['end'] - interval['begin']
        #data['spectator_mode'] = max(count_dict, key=lambda x: count_dict[x])
        #count_dict = Counter()
        # for interval in new_status['film_format']:
        #    if interval['status'] != 'n/a':
        #        count_dict[interval['status']] += interval['end'] - interval['begin']
        # data['film_format'] = max(count_dict, key=lambda x: count_dict[x])
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
        if not rounds:  # FIXME maybe
            for i, interval in enumerate(new_status['game']):
                if interval != 'not_game':
                    beg = interval['begin']
                    rounds = [{'begin': beg, 'end': beg + 120, 'pauses': [], 'replays': [],
                               'smaller_windows': [], 'left_zooms': [], 'right_zooms': [], 'left_color': 'red',
                               'right_color': 'blue', 'spectator_mode': 'original', 'attacking_side': 'N',
                               'map': 'oasis'}]
        return data, rounds
