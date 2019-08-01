import os
import torch
import numpy as np
import cv2
from annotator.annotator.classes.base import BaseAnnotator
from torch.autograd import Variable

from annotator.models.cnn import StatusCNN
from annotator.training.helper import load_set
from annotator.training.ctc_helper import loadData
from collections import defaultdict, Counter
from annotator.config import sides, BOX_PARAMETERS


class Team(object):
    def __init__(self, side, player_states):
        self.side = side
        self.player_states = player_states
        self.color = max(Counter([x.color for x in player_states.values()]).items(), key=lambda x: x[1])[0]

    def heroes_at_time(self, time_point):
        heroes = []
        for k, v in self.player_states.items():
            heroes.append(v.hero_at_time(time_point))
        return heroes

    def has_hero_at_time(self, hero, time_point):
        for k, v in self.player_states.items():
            if v.hero_at_time(time_point) == hero:
                return True
        return False

    def alive_at_time(self, hero, time_point):
        for k, v in self.player_states.items():
            if v.hero_at_time(time_point) == hero:
                return v.alive_at_time(time_point)
        return None

    def get_death_events(self):
        death_events = []
        for k, v in self.player_states.items():
            death_events.extend(v.generate_death_events())
        return death_events


class PlayerState(object):
    def __init__(self, player, statuses):
        self.player = player
        self.statuses = statuses
        self.name = statuses['player_name']
        self.color = self.statuses['color'][0]['status']

    def hero_at_time(self, time_point):
        for hero_state in self.statuses['hero']:
            if hero_state['end'] >= time_point >= hero_state['begin']:
                return hero_state['status']

    def alive_at_time(self, time_point):
        for state in self.statuses['alive']:
            if state['end'] >= time_point >= state['begin']:
                return state['status'] == 'alive'
        return None

    def generate_death_events(self):
        deaths = []
        for alive_state in self.statuses['alive']:
            if alive_state['status'] == 'dead':
                deaths.append(
                    {'time_point': alive_state['begin'] + 0.3, 'hero': self.hero_at_time(alive_state['begin']),
                     'color': self.color, 'player': self.player})
        return deaths

    def generate_switches(self):
        switches = []
        for hero_state in self.statuses['hero']:
            switches.append([hero_state['begin'], hero_state['status']])
        return switches

    def generate_status_effects(self, enemy_team, friendly_team):
        status_effect_types = ['antiheal', 'asleep', 'frozen', 'hacked', 'stunned', 'immortal']
        statuses = {}
        print(self.statuses['status'])
        for stype in status_effect_types:
            statuses[stype] = []
            if stype in self.statuses:
                for interval in self.statuses[stype]:
                    if interval['status'] == 'antiheal':
                        if not enemy_team.has_hero_at_time('ana', interval['begin']):
                            continue
                    if interval['status'] == 'immortal':
                        if not friendly_team.has_hero_at_time('baptiste', interval['begin']):
                            continue
                    if not interval['status'].startswith('not_') and self.alive_at_time(interval['begin']):
                        statuses[stype].append(interval)

        for interval in self.statuses['status']:
            if interval['status'] == 'asleep':
                if not enemy_team.has_hero_at_time('ana', interval['begin']):
                    continue
            if interval['status'] == 'hacked':
                if not enemy_team.has_hero_at_time('sombra', interval['begin']):
                    continue
            if interval['status'] == 'frozen':
                if not enemy_team.has_hero_at_time('mei', interval['begin']):
                    continue
            if interval['status'] != 'normal' and self.alive_at_time(interval['begin']):
                statuses[interval['status']].append(interval)
        return statuses

    def generate_ults(self, mech_deaths=None, revive_events=None):
        print(self.player)
        if mech_deaths is None:
            mech_deaths = []
        else:
            mech_deaths = [x['time_point'] - 1 for x in mech_deaths if x['color'] == self.color]
        if revive_events is None:
            revive_events = []
        # Possible dva states:
        # Not-dva
        # In mech - alive
        # Out of mech - alive
        # Dead
        switches = self.generate_switches()
        dva_notdva = []
        mech_states = []
        for s in self.statuses['hero']:
            print(s)
            if s['status'] == 'd.va':
                current_time = s['begin']
                current_state = 'in_mech'
                while current_time < s['end']:
                    print(current_time)
                    mech_state = {'begin': current_time, 'status':current_state}
                    if current_state == 'in_mech':
                        for d in mech_deaths:
                            if d >= current_time:
                                nearest_mech_death = d
                                break
                        else:
                            nearest_mech_death = 100000
                        for u in self.statuses['ult']:
                            if u['begin'] <= current_time:
                                continue
                            if u['status'] == 'no_ult':
                                nearest_ult_use = u['begin']
                                break
                        else:
                            nearest_ult_use = 100000
                        if nearest_mech_death < nearest_ult_use and nearest_mech_death < s['end']:
                            mech_state['end'] = nearest_mech_death
                        elif nearest_ult_use < nearest_mech_death and nearest_ult_use < s['end']:
                            mech_state['end'] = nearest_ult_use
                        else:
                            mech_state['end'] = s['end']
                        current_state = 'out_of_mech'
                    elif current_state == 'out_of_mech':
                        for d in self.statuses['alive']:
                            if d['begin'] >= current_time and d['status'] == 'dead':
                                nearest_death = d['begin']
                                break
                        else:
                            nearest_death = 100000
                        for u in self.statuses['ult']:
                            if u['begin'] <= current_time:
                                continue
                            if u['status'] == 'no_ult':
                                nearest_ult_use = u['begin']
                                break
                        else:
                            nearest_ult_use = 100000
                        if nearest_death < nearest_ult_use and nearest_death < s['end']:
                            mech_state['end'] = nearest_death
                            current_state = 'dead'
                        elif nearest_ult_use < nearest_death and nearest_ult_use < s['end']:
                            mech_state['end'] = nearest_ult_use
                            current_state = 'in_mech'
                        else:
                            mech_state['end'] = s['end']
                    elif current_state == 'dead':
                        for d in self.statuses['alive']:
                            if d['begin'] == current_time and d['status'] == 'dead':
                                mech_state['end'] = d['end']
                                if d['end'] - d['begin'] < 10: # Revive baby dva
                                    current_state = 'out_of_mech'
                                else:
                                    current_state = 'in_mech'
                                break
                    print(mech_state)
                    mech_states.append(mech_state)
                    current_time = mech_state['end']
        print('alive', self.statuses['alive'])
        print('ult', self.statuses['ult'])
        print('mech_deaths', mech_deaths)
        print('mech_states', mech_states)
        ult_gains, ult_uses, ult_ends = [], [], []
        end_time = self.statuses['alive'][-1]['end']
        for i, ult_state in enumerate(self.statuses['ult']):
            if i > 0 and ult_state['status'] == 'no_ult' and ult_state['end'] - ult_state['begin'] < 0.3 and ult_state['end'] != end_time:
                continue
            if False and self.hero_at_time(ult_state['begin']) == 'd.va':
                for ms in mech_states:
                    if ms['begin'] <= ult_state['begin'] <= ms['end'] and ms['status'] != 'out_of_mech':
                        if ult_state['status'] == 'has_ult':
                            add = True
                            if len(ult_gains) > 0:
                                last_ug = ult_gains[-1]
                                if len(ult_uses) > 0:
                                    last_uu = ult_uses[-1]
                                    if last_uu < last_ug:
                                        add = False
                            print(add, ult_state)
                            if add:
                                ult_gains.append(ult_state['begin'])
                        else:
                            add = True
                            for md in mech_deaths:
                                if abs(md - ult_state['begin']) < 5:
                                    add = False
                                    break
                            else:
                                if len(ult_gains) == 0:
                                    add = False
                                elif len(ult_uses) > 0:
                                    last_ug = ult_gains[-1]
                                    last_uu = ult_uses[-1]
                                    if last_uu > last_ug:
                                        add = False
                            if add:
                                ult_uses.append(ult_state['begin'])
                        break
            else:
                if ult_state['status'] == 'has_ult':
                    if self.statuses['ult'][i-1]['status'] == 'using_ult':
                        ult_gains.append(self.statuses['ult'][i-1]['begin'])
                    else:

                        ult_gains.append(ult_state['begin'])
                elif ult_state['status'] == 'using_ult' and self.statuses['ult'][i-1]['status'] == 'has_ult':
                    ult_uses.append(ult_state['begin'])
                    ult_ends.append(ult_state['end'])
        print('ult_gains', ult_gains)
        print('ult_uses', ult_uses)
        print('ult_ends', ult_ends)
        return ult_gains, ult_uses, ult_ends


class PlayerStatusAnnotator(BaseAnnotator):
    time_step = 0.1
    batch_size = 200
    identifier = 'player_status'
    box_settings = 'LEFT'

    def __init__(self, film_format, model_directory, device, left_color, right_color, player_names):
        super(PlayerStatusAnnotator, self).__init__(film_format, device)
        self.figure_slot_params(film_format)
        self.model_directory = model_directory
        self.left_team_color = left_color
        self.right_team_color = right_color
        self.player_names = player_names
        set_paths = {
            'hero': os.path.join(model_directory, 'hero_set.txt'),
            'alive': os.path.join(model_directory, 'alive_set.txt'),
            'ult': os.path.join(model_directory, 'ult_set.txt'),
            'status': os.path.join(model_directory, 'status_set.txt'),
            'antiheal': os.path.join(model_directory, 'antiheal_set.txt'),
            'immortal': os.path.join(model_directory, 'immortal_set.txt'),
            'color': os.path.join(model_directory, 'color_set.txt'),

        }

        sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)
        self.model = StatusCNN(sets)
        self.model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pth')))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)
        self.to_predict = {}
        self.statuses = {}
        self.shape = (self.batch_size, 3, int(self.params['HEIGHT'] * self.resize_factor),
                      int(self.params['WIDTH'] * self.resize_factor))
        self.images = Variable(torch.FloatTensor(self.batch_size, 3, int(self.params['HEIGHT'] * self.resize_factor),
                      int(self.params['WIDTH'] * self.resize_factor)).to(device))

        for s in self.slot_params.keys():
            self.to_predict[s] = np.zeros(self.shape, dtype=np.uint8)
            self.statuses[s] = {k: [] for k in list(self.model.sets.keys())}

    def figure_slot_params(self, film_format):
        left_params = BOX_PARAMETERS[film_format]['LEFT']
        right_params = BOX_PARAMETERS[film_format]['RIGHT']
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

    def process_frame(self, frame):
        #cv2.imshow('frame', frame)
        for s, params in self.slot_params.items():

            x = params['x']
            y = params['y']
            box = frame[y: y + self.params['HEIGHT'],
                  x: x + self.params['WIDTH']]
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
            self.begin_time += self.batch_size * self.time_step

    def annotate(self):
        import time
        begin = time.time()
        if self.process_index == 0:
            return
        for s in self.slot_params.keys():
            #print(s)
            b = time.time()
            loadData(self.images, torch.from_numpy(self.to_predict[s]).float())
            predicteds = self.model({'image': self.images})
            #print('got predictions:', time.time()-b)
            b = time.time()
            for k, v in predicteds.items():
                #print(k)
                #print(predicteds[k])
                _, predicteds[k] = torch.max(v.to('cpu'), 1)


                for t_ind in range(self.process_index - 1):
                    #cv2.imshow('frame_{}'.format(t_ind), np.transpose(self.to_predict[s][t_ind], axes=(1, 2, 0)))
                    current_time = self.begin_time + (t_ind * self.time_step)
                    #print(current_time)
                    label = self.model.sets[k][predicteds[k][t_ind]]
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

    def generate_statuses(self):
        statuses = {}
        for s, status_dict in self.statuses.items():
            print(s)
            statuses[s] = {}
            for k, v in status_dict.items():
                print(k)
                if k == 'color':
                    new_v = defaultdict(float)
                    begin =v[0]['begin']
                    end = v[-1]['end']
                    for i in v:
                        new_v[i['status']] += i['end'] - i['begin']
                    new_v = max(new_v.keys(), key=lambda x: new_v[x])
                    statuses[s][k] = [{'begin': begin, 'end': end, 'status':new_v}]
                else:
                    if k in ['hero', 'alive']:
                        threshold = 1
                    elif k in ['ult']:
                        threshold = 0.5
                    else:
                        threshold = 0.3
                    v = [x for x in v if x['end'] - x['begin'] > threshold]
                    new_v = []
                    for x in v:
                        if not new_v:
                            new_v.append(x)
                        else:
                            if x['status'] == new_v[-1]['status']:
                                new_v[-1]['end'] = x['end']
                            else:
                                new_v.append(x)
                    statuses[s][k] = new_v
        print(statuses)
        return statuses

    def generate_teams(self):
        teams = {'left': {}, 'right': {}}
        for k, status_dict in self.statuses.items():
            side, ind = k
            new_statuses = {}
            new_statuses['player_name'] = self.player_names[k]
            if side == 'left':
                new_statuses['color'] = self.left_team_color
            else:
                new_statuses['color'] = self.right_team_color

            for k, v in status_dict.items():
                if k == 'color':
                    new_v = defaultdict(float)
                    begin =v[0]['begin']
                    end = v[-1]['end']
                    for i in v:
                        new_v[i['status']] += i['end'] - i['begin']
                    new_v = max(new_v.keys(), key=lambda x: new_v[x])
                    new_statuses[k] = [{'begin': begin, 'end': end, 'status':new_v}]
                else:
                    if k in ['hero', 'alive']:
                        threshold = 1
                    else:
                        threshold = 0.3
                    v = [x for x in v if x['end'] - x['begin'] > threshold]
                    new_v = []
                    for i,x in enumerate(v):
                        if k == 'ult' and x['status'] != 'no_ult' and x['begin'] < 10:
                            continue
                        if not new_v:
                            new_v.append(x)
                        else:
                            if x['status'] == new_v[-1]['status']:
                                new_v[-1]['end'] = x['end']
                            else:
                                if k == 'ult' and x['status'] == 'using_ult' and i != len(v) - 1 and v[i+1]['status'] == 'has_ult':
                                    continue
                                new_v.append(x)
                    new_v[0]['begin'] = 0
                    new_statuses[k] = new_v
            death_events = []
            revive_events = []
            for i, interval in enumerate(new_statuses['alive']):
                if interval['status'] == 'dead':
                    death_events.append(interval['begin'])
                    if interval['end'] - interval['begin'] <= 9.8 and i < len(new_statuses['alive']) - 1:
                        revive_events.append(interval['end'])
            new_statuses['death_events'] = death_events
            new_statuses['revive_events'] = revive_events
            for k2, v2 in new_statuses.items():
                print(k2, v2)
            teams[side][ind] = PlayerState(k, new_statuses)
        left_team = Team('left', teams['left'])
        left_team.color = self.left_team_color
        for p, v in left_team.player_states.items():
            v.color = self.left_team_color
        right_team = Team('right', teams['right'])
        right_team.color = self.right_team_color
        for p, v in right_team.player_states.items():
            v.color = self.right_team_color
        return left_team, right_team