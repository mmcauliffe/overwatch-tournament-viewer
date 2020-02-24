import os
import torch
import numpy as np
import cv2
import h5py
from annotator.annotator.classes.base import BaseAnnotator
from torch.autograd import Variable

from annotator.models.cnn import StatusCNN
from annotator.training.helper import load_set
from annotator.training.ctc_helper import loadData
from collections import defaultdict, Counter
from annotator.config import sides, BOX_PARAMETERS
from annotator.annotator.classes.hmm import HMM
from annotator.api_requests import get_round_states
from annotator.annotator.classes.base import filter_statuses, coalesce_statuses


def check_time_in_ranges(time_point, ranges):
    for r in ranges:
        if r['begin'] <= time_point <= r['end']:
            return True
    return False

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

    def get_hero_range(self, heroes):
        playtimes = []
        final_time = 0
        for k, v in self.player_states.items():
            for hero_state in v.statuses['hero']:
                if hero_state['status'] in heroes:
                    playtimes.append(hero_state)
                if hero_state['end'] > final_time:
                    final_time = hero_state['end']
        merged_playtimes = []
        for s in sorted(playtimes, key=lambda x: x['begin']):
            if not merged_playtimes:
                merged_playtimes.append({'begin': s['begin'], 'end': s['end']})
                continue
            if s['begin'] <= merged_playtimes[-1]['end'] and s['end'] > merged_playtimes[-1]['end']:
                merged_playtimes[-1]['end'] = s['end']
            elif s['begin'] > merged_playtimes[-1]['end']:
                merged_playtimes.append({'begin': s['begin'], 'end': s['end']})
        return merged_playtimes


class PlayerState(object):
    def __init__(self, player, statuses, color=None):
        self.player = player
        self.statuses = statuses
        self.name = statuses['player_name']
        self.color = color
        self.side = player[0]

    def hero_at_time(self, time_point):
        for hero_state in self.statuses['hero']:
            if hero_state['end'] >= time_point >= hero_state['begin']:
                return hero_state['status']

    def played_dva(self):
        for hero_state in self.statuses['hero']:
            if hero_state['status'] == 'd.va':
                return True
        return False

    def dva_ranges(self):
        ranges = []
        for hero_state in self.statuses['hero']:
            if hero_state['status'] == 'd.va':
                ranges.append(hero_state)
        return ranges


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
                     'side': self.side, 'player': self.player})
        return deaths

    def generate_switches(self):
        switches = []
        for i, hero_state in enumerate(self.statuses['hero']):
            print(self.player, hero_state)
            begin = hero_state['begin']
            end = hero_state['end']
            if i > 0 and begin - self.statuses['hero'][i - 1]['end'] < 5:
                begin = self.statuses['hero'][i - 1]['end'] + 0.1
            alive_check = True
            for alive_state in self.statuses['alive']:
                if alive_state['begin'] <= hero_state['begin'] and\
                        hero_state['end'] <= alive_state['end'] and \
                        alive_state['status'] == 'dead':
                    alive_check = False
                    break
            status_check = True
            for status_state in self.statuses['status']:
                if status_state['begin'] <= hero_state['begin'] and\
                        hero_state['end'] <= status_state['end'] and \
                        status_state['status'] != 'normal':
                    status_check = False
                    break
            if alive_check and status_check:
                if not switches or switches[-1]['hero'] != hero_state['status']:
                    switches.append({'begin': begin, 'hero': hero_state['status'], 'end': end})
        return switches

    def generate_status_effects(self, enemy_team, friendly_team):
        import time
        stunners = ['brigitte', 'reinhardt', 'roadhog', 'doomfist', 'mccree', 'sigma']
        status_effect_types = ['antiheal', 'asleep', 'frozen', 'hacked', 'stunned', 'immortal', 'discord',
                               'resurrecting', 'nanoboosted']
        statuses = {}
        print(self.statuses['status'])
        for stype in status_effect_types:
            statuses[stype] = []
            if stype in self.statuses:
                for interval in self.statuses[stype]:
                    if interval['status'] == 'antiheal':
                        if not enemy_team.has_hero_at_time('ana', interval['begin']):
                            continue
                        if interval['end'] - interval['begin'] > 4:
                            interval['end'] = interval['begin'] + 3.9
                    if interval['status'] == 'immortal':
                        if not friendly_team.has_hero_at_time('baptiste', interval['begin']):
                            continue
                    if interval['status'] == 'nanoboosted':
                        if not friendly_team.has_hero_at_time('ana', interval['begin']):
                            continue
                    if not interval['status'].startswith('not_') and self.alive_at_time(interval['begin']):
                        statuses[stype].append({'begin': interval['begin'], 'end': round(interval['end'] + 0.1, 1)})

        for i, interval in enumerate(self.statuses['status']):
            if interval['status'] == 'asleep':
                if not enemy_team.has_hero_at_time('ana', interval['begin']):
                    continue
                if not self.alive_at_time(round(interval['begin'] - 0.5, 1)):
                    continue
            if interval['status'] == 'hacked':
                if not enemy_team.has_hero_at_time('sombra', interval['begin']):
                    continue
            if interval['status'] == 'frozen':
                if not enemy_team.has_hero_at_time('mei', interval['begin']):
                    continue
            if interval['status'] == 'discord':
                if not enemy_team.has_hero_at_time('zenyatta', interval['begin']):
                    continue
            if interval['status'] == 'resurrecting':
                if not friendly_team.has_hero_at_time('mercy', interval['begin']):
                    continue
            if interval['status'] == 'stunned':
                has_no_stunner = True
                for stunner in stunners:
                    if enemy_team.has_hero_at_time(stunner, interval['begin']):
                        has_no_stunner = False
                        break
                if has_no_stunner and self.hero_at_time(interval['begin']) != 'd.va':
                    continue
            if interval['status'] != 'normal':  # and self.alive_at_time(interval['begin']):
                beg = interval['begin']
                if i != 0:
                    beg = self.statuses['status'][i - 1]['end'] + 0.1
                statuses[interval['status']].append({'begin': beg, 'end': interval['end'] + 0.1})
        return statuses

    def generate_ults(self, mech_deaths=None, revive_events=None, sequences= None):
        print(self.player)
        if sequences is None:
            sequences = []
        if mech_deaths is None:
            mech_deaths = []
        else:
            mech_deaths = [x['time_point'] - 1 for x in mech_deaths if x['side'] == self.side]
        if revive_events is None:
            revive_events = []
        else:
            revive_events = [x['time_point'] - 1 for x in revive_events if x['side'] == self.side]
        print(revive_events)
        # Possible dva states:
        # Not-dva
        # In mech - alive
        # Out of mech - alive
        # Dead
        dva_notdva = []
        mech_states = []
        for s in self.statuses['hero']:
            print(s)
            if s['status'] == 'd.va':
                current_time = s['begin']
                current_state = 'in_mech'
                while current_time < s['end']:
                    print(current_time)
                    mech_state = {'begin': current_time, 'status': current_state}
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
                            if u['status'] == 'using_ult':
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
                        for i, u in enumerate(self.statuses['ult']):
                            if u['begin'] < current_time:
                                continue
                            if u['status'] == 'using_ult':
                                nearest_ult_use = u['end']
                                break
                            if u['status'] == 'has_ult' and self.statuses['ult'][i-1]['status'] == 'no_ult' and \
                                    u['begin'] - self.statuses['ult'][i-1]['end'] > 1 and \
                                    self.statuses['ult'][i-1]['end'] != mech_state['begin']:
                                nearest_ult_use = self.statuses['ult'][i-1]['end']
                                break

                        else:
                            nearest_ult_use = 100000
                        if nearest_death < nearest_ult_use and nearest_death < s['end']:
                            mech_state['end'] = nearest_death
                            current_state = 'dead'
                        elif nearest_ult_use < nearest_death and nearest_ult_use <= s['end']:
                            mech_state['end'] = round(nearest_ult_use + 0.1, 1)
                            current_state = 'in_mech'
                        else:
                            mech_state['end'] = s['end']
                        if mech_state['end'] - mech_state['begin'] > 30:
                            current_state = 'in_mech'
                            mech_state['end'] = round(mech_state['begin'] + 30, 1)
                    elif current_state == 'dead':
                        for i, d in enumerate(self.statuses['alive']):
                            if d['begin'] == current_time and d['status'] == 'dead':
                                mech_state['end'] = d['end']
                        was_revived = False
                        for re in revive_events:
                            if re - mech_state['end'] < 5:
                                was_revived = True
                                break
                        if was_revived:
                            current_state = 'out_of_mech'
                        else:
                            current_state = 'in_mech'
                    print(mech_state)
                    mech_states.append(mech_state)
                    current_time = mech_state['end']
        print('alive', self.statuses['alive'])
        print('ult', self.statuses['ult'])
        print('mech_deaths', mech_deaths)
        print('mech_states', mech_states)
        ultimates = []
        end_time = self.statuses['alive'][-1]['end']
        for i, ult_state in enumerate(self.statuses['ult']):

            if ult_state['status'] == 'has_ult':
                gained = ult_state['begin']
                if i > 0 and self.statuses['ult'][i - 1]['status'] == 'no_ult' and \
                        ult_state['begin'] -self.statuses['ult'][i - 1]['end'] < 3:
                    gained = round(self.statuses['ult'][i - 1]['end'] + 0.1, 1)
                ultimate = {'gained': gained}
                if i < len(self.statuses['ult']) - 1:
                    if self.statuses['ult'][i + 1]['status'] == 'using_ult':
                        ultimate['used'] = round(ult_state['end'] + 0.1, 1)
                        ultimate['ended'] = round(self.statuses['ult'][i + 1]['end'] + 0.1, 1)
                    elif self.statuses['ult'][i + 1]['status'] == 'no_ult' and self.hero_at_time(ult_state['end']) != 'd.va'\
                            and self.hero_at_time(ult_state['end']) == self.hero_at_time(ult_state['end'] + 2):
                        for s in sequences:
                            if self.statuses['ult'][i + 1]['begin'] - s[0] < 0:
                                continue
                            if self.statuses['ult'][i + 1]['begin'] - s[0] < 1:
                                ultimate['used'] = round(self.statuses['ult'][i + 1]['begin'] - 1.0, 1)
                                ultimate['ended'] = round(self.statuses['ult'][i + 1]['begin'] + 0.1, 1)
                                break
                        else:
                            if self.statuses['ult'][i + 1]['end'] - self.statuses['ult'][i + 1]['begin'] > 10:
                                ultimate['used'] = round(ult_state['end'] + 0.1, 1)
                                ultimate['ended'] = round(self.statuses['ult'][i + 1]['begin'] + 0.1, 1)

            elif ult_state['status'] == 'using_ult' and i > 0 and self.statuses['ult'][i - 1]['status'] == 'no_ult':
                if ult_state['begin'] -self.statuses['ult'][i - 1]['end'] < 3:
                    gained = round(self.statuses['ult'][i - 1]['end'] + 0.1, 1)
                else:
                    gained = round(ult_state['begin'] - 0.1, 1)
                ultimate = {'gained': gained,
                            'used': ult_state['begin'],
                            'ended': round(ult_state['end'] + 0.1, 1)}
            else:
                continue
            if False and mech_states:  # Ignore dva states for now
                out_of_mech_check = False
                print(ultimate)
                for i, ms in enumerate(mech_states):
                    if ms['status'] == 'out_of_mech' and ms['begin'] <= ultimate['gained'] < ms['end']:
                        if 'used' in ultimate and ultimate['used'] > ms['end']:
                            break
                        out_of_mech_check = True
                        break
                    elif i != 0 and mech_states[i - 1] == 'dead' and ms['begin'] <= ultimate['gained'] <= ms[
                        'begin'] + 0.2:
                        out_of_mech_check = True
                        break
                print(out_of_mech_check)
                if not out_of_mech_check:
                    for h in self.statuses['hero']:
                        if h['begin'] > ultimate['gained']:
                            break
                        cur = h
                    print(not ultimates, not ultimates or 'used' in ultimates[-1], not ultimates or cur['begin'] > ultimates[-1]['gained'])
                    if not ultimates or 'used' in ultimates[-1] or cur['begin'] > ultimates[-1]['gained']:
                        if ultimates and 'ended' in ultimates[-1] and ultimate['gained'] - ultimates[-1]['ended'] < 5:
                            continue
                        ultimates.append(ultimate)
                    elif 'used' in ultimate:
                        ultimates[-1]['used'] = ultimate['used']
                        ultimates[-1]['ended'] = ultimate['ended']
            else:
                if ultimates and self.hero_at_time(ultimate['gained']) == 'd.va':
                    if 'used' in ultimate and ultimate['used'] - ultimate['gained'] < 2 and not self.alive_at_time(ultimate['gained']-0.5):
                        continue
                    if 'ended' in ultimates[-1] and ultimate['gained'] - 2 < ultimates[-1]['ended']:
                        continue
                    if 'used' not in ultimates[-1] and 'used' in ultimate and \
                            self.hero_at_time(ultimates[-1]['gained']) == 'd.va' and \
                            self.hero_at_time(ultimate['used']):
                        ultimates[-1]['used'] = ultimate['used']
                        ultimates[-1]['ended'] = ultimate['ended']
                        continue
                ultimates.append(ultimate)
        print(ultimates)
        if mech_states:
            print('DVA ANALYSIS', self.player)
            print('alive', self.statuses['alive'])
            print('mech_states', mech_states)
            print(ultimates)
            # if self.side == 'right':
            #    error
        return ultimates


class PlayerStatusAnnotator(BaseAnnotator):
    time_step = 0.1
    batch_size = 100
    identifier = 'player_status'
    box_settings = 'LEFT'

    def __init__(self, film_format, model_directory, device, left_color, right_color, player_names, spectator_mode='O',
                 debug=False):
        super(PlayerStatusAnnotator, self).__init__(film_format, device, debug=debug)
        print('=== SETTING UP PLAYER STATUS ANNOTATOR ===')
        if debug:
            self.batch_size = 1
        else:
            self.batch_size = 100
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
            'nanoboosted': os.path.join(model_directory, 'nanoboosted_set.txt'),
        }
        sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)
        input_set_files = {
            # 'color': os.path.join(model_directory, 'color_set.txt'),
            # 'enemy_color': os.path.join(model_directory, 'enemy_color_set.txt'),
            #'spectator_mode': os.path.join(model_directory, 'spectator_mode_set.txt'),
        }
        input_sets = {}
        for k, v in input_set_files.items():
            input_sets[k] = load_set(v)
        self.model = StatusCNN(sets, input_sets)
        spec_dir = os.path.join(model_directory, self.spectator_mode)
        if os.path.exists(spec_dir):
            model_path = os.path.join(spec_dir, 'model.pth')
            print('Using {} status model!'.format(self.spectator_mode))
        else:
            model_path = os.path.join(model_directory, 'model.pth')
            print('Using base status model!')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)
        # prob_path = os.path.join(model_directory, 'hmm_probs.h5')
        # self.hmms = {}
        # with h5py.File(prob_path, 'r') as hf5:
        #    for k in ['hero', 'ult', 'status']:
        #        self.hmms[k] = HMM(len(sets[k]))
        #        self.hmms[k].startprob_ = hf5['{}_init'.format(k)][:].astype(np.float_)
        #        trans = hf5['{}_trans'.format(k)][:].astype(np.float_)
        #        for i in range(trans.shape[0]):
        #            if trans[i, i] == 0:
        #                trans[i, i] = 1
        #        self.hmms[k].transmat_ = trans
        self.statuses = {}
        # self.probs = {}
        self.image_height = int(self.params['HEIGHT'] * self.resize_factor)
        self.image_width = int(self.params['WIDTH'] * self.resize_factor)
        self.shape = (12, self.batch_size, 3, self.image_height, self.image_width)
        self.images = Variable(
            torch.FloatTensor(12 * self.batch_size, 3, self.image_height, self.image_width).to(device))
        self.inputs = {}

        self.inputs['color'] = Variable(torch.FloatTensor(12, self.batch_size, 3).to(device))
        for i in range(3):
            self.inputs['color'][:6, :, i] = self.left_team_color[i]
            self.inputs['color'][6:, :, i] = self.right_team_color[i]
        self.inputs['color'] = self.inputs['color'].view(12 * self.batch_size, 3)
        self.inputs['color'] = ((self.inputs['color'] / 255) - 0.5) / 0.5

        self.inputs['enemy_color'] = Variable(torch.FloatTensor(12, self.batch_size, 3).to(device))
        for i in range(3):
            self.inputs['enemy_color'][:6, :, i] = self.right_team_color[i]
            self.inputs['enemy_color'][6:, :, i] = self.left_team_color[i]
        self.inputs['enemy_color'] = self.inputs['enemy_color'].view(12 * self.batch_size, 3)
        self.inputs['enemy_color'] = ((self.inputs['enemy_color'] / 255) - 0.5) / 0.5

        self.to_predict = np.zeros(self.shape, dtype=np.uint8)
        self.ignored_time_points = {}
        self.status_effect_probs = {}
        self.stored_probs = {}
        for s in self.slot_params.keys():
            self.status_effect_probs[s] = []
            # self.probs[s] = {k: [] for k in list(self.model.sets.keys())}
            self.statuses[s] = {k: [] for k in list(self.model.sets.keys()) if k != 'status'}
            self.ignored_time_points[s] = []

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
        if 'ZOOMED_LEFT' in BOX_PARAMETERS[film_format]:
            self.zoomed_width = BOX_PARAMETERS[film_format]['ZOOMED_LEFT']['WIDTH']
            self.zoomed_height = BOX_PARAMETERS[film_format]['ZOOMED_LEFT']['HEIGHT']
            zoomed_left = BOX_PARAMETERS[film_format]['ZOOMED_LEFT']
            zoomed_right = BOX_PARAMETERS[film_format]['ZOOMED_RIGHT']
            self.zoomed_params = {}
            for side in sides:
                if side == 'left':
                    p = zoomed_left
                else:
                    p = zoomed_right
                for i in range(6):
                    self.zoomed_params[(side, i)] = {}
                    self.zoomed_params[(side, i)]['x'] = p['X'] + (p['WIDTH'] + p['MARGIN']) * i
                    self.zoomed_params[(side, i)]['y'] = p['Y']
        else:
            self.zoomed_params = self.slot_params
        print(self.slot_params)

    def get_zooms(self, r):
        round_states = get_round_states(r['id'])
        self.zooms = {'left': [], 'right': []}
        zooms = round_states['zoomed_bars']
        for side in self.zooms.keys():
            for z in zooms[side]:
                if z['status'] == 'zoomed':
                    self.zooms[side].append(z)

    def is_zoomed(self, time_point, side):
        for z in self.zooms[side]:
            if round(z['begin'] + 0.4, 1) <= time_point < round(z['end'] - 0.4, 1):
                return True
        return False

    def ignore_time_point(self, time_point, side):
        for z in self.zooms[side]:
            if z['begin'] <= time_point < round(z['begin'] + 0.4, 1):
                return True
            if round(z['end'] - 0.4, 1) <= time_point <= z['end']:
                return True
        return False

    def process_frame(self, frame, time_point):
        # cv2.imshow('frame', frame)
        for i, (s, params) in enumerate(self.slot_params.items()):
            side = s[0]
            zoomed = self.is_zoomed(time_point, side)
            if self.ignore_time_point(time_point, side):
                self.ignored_time_points[s].append(time_point)
            if zoomed:
                params = self.zoomed_params[s]
            else:
                params = self.slot_params[s]

            x = params['x']
            y = params['y']
            if zoomed:
                box = frame[y: y + self.zoomed_height,
                      x: x + self.zoomed_width]
                box = cv2.resize(box, (self.image_height, self.image_width))

            else:
                box = frame[y: y + self.params['HEIGHT'],
                      x: x + self.params['WIDTH']]
            if self.debug:
                cv2.imshow('frame_{}_{}'.format(*s), box)
                print('IGNORED', s, self.ignored_time_points[s])

            box = np.transpose(box, axes=(2, 0, 1))
            self.to_predict[i, self.process_index, ...] = box[None]
        self.process_index += 1

        if self.process_index == self.batch_size:
            self.annotate()

    def reset(self, begin_time):
        if self.process_index:
            self.stored_probs[self.begin_time] = {}
            for s in self.slot_params.keys():
                self.stored_probs[self.begin_time][s] = np.array(self.status_effect_probs[s])
                self.status_effect_probs[s] = []
        self.to_predict = np.zeros(self.shape, dtype=np.uint8)
        self.process_index = 0
        self.begin_time = begin_time

    def annotate(self):
        import time
        begin = time.time()
        if self.process_index == 0:
            return
        # print(s)
        b = time.time()
        t = torch.from_numpy(self.to_predict).float()
        t = ((t / 255) - 0.5) / 0.5
        loadData(self.images, t.view(12 * self.batch_size, 3, self.image_height, self.image_width))
        ins = {'image': self.images,
               'color': self.inputs['color'], 'enemy_color': self.inputs['enemy_color']}
        with torch.no_grad():
            predicteds = self.model(ins)
        # print('got predictions:', time.time()-b)
        b = time.time()
        for k, v in predicteds.items():
            # print(k)
            # print(predicteds[k])
            predicteds[k] = predicteds[k].view(12, self.batch_size, -1).to('cpu')
            for i, s in enumerate(self.slot_params.keys()):
                v = predicteds[k][i, :self.process_index, ...]
                if k == 'status':
                    self.status_effect_probs[s].extend(v.numpy())
                    continue
                # self.probs[s][k].extend(v.numpy())
                _, v = torch.max(v, 1)

                for t_ind in range(self.process_index):
                    # cv2.imshow('frame_{}'.format(t_ind), np.transpose(self.to_predict[s][t_ind], axes=(1, 2, 0)))
                    current_time = round(self.begin_time + (t_ind * self.time_step), 1)
                    if current_time in self.ignored_time_points[s]:
                        continue
                    label = self.model.sets[k][v[t_ind]]
                    # print(label)
                    if len(self.statuses[s][k]) == 0:
                        self.statuses[s][k].append({'begin': 0, 'end': 0, 'status': label})
                    else:
                        if label == self.statuses[s][k][-1]['status']:
                            self.statuses[s][k][-1]['end'] = current_time
                        else:
                            self.statuses[s][k].append(
                                {'begin': current_time, 'end': current_time, 'status': label})
        if self.debug:
            for s in self.slot_params.keys():
                print(s)
                for k, v in self.statuses[s].items():
                    print('   ', k, v[-1])
            # cv2.waitKey()
        self.reset(self.begin_time + (self.batch_size * self.time_step))
        print('Status annotate took: ', time.time() - begin)

    def generate_teams(self):
        teams = {'left': {}, 'right': {}}
        colors = {'left': self.left_team_color, 'right': self.right_team_color}

        # for s, probs in self.probs.items():
        #    for k, hmm in self.hmms.items():

        #        self.statuses[s][k] = []
        #        p = np.array(probs[k]).astype(np.float_)
        #        log, z = hmm.decode(p)
        #        for i, z1 in enumerate(z):
        #            current_time = i * self.time_step
        #            label = self.model.sets[k][z1]
        #            if len(self.statuses[s][k]) == 0:
        #                self.statuses[s][k].append({'begin': 0, 'end': 0, 'status': label})
        #            else:
        #                if label == self.statuses[s][k][-1]['status']:
        #                    self.statuses[s][k][-1]['end'] = current_time
        #                else:
        #                    self.statuses[s][k].append(
        #                        {'begin': current_time, 'end': current_time, 'status': label})
        for slot, status_dict in self.statuses.items():
            side, ind = slot
            new_statuses = {}
            new_statuses['player_name'] = self.player_names[slot]
            if side == 'left':
                new_statuses['color'] = self.left_team_color
            else:
                new_statuses['color'] = self.right_team_color
            print(slot)
            print('IGNORED', self.ignored_time_points[slot])
            for k, v in sorted(status_dict.items()):
                print(k)
                print(v)
                v = filter_statuses(v, 0)
                if k == 'hero':
                    new_v = []
                    threshold = 0.5
                    for i, x in enumerate(v):
                        if not new_v:
                            new_v.append(x)
                        else:
                            # No switches while dead
                            alive_check = True
                            # for x2 in status_dict['alive']:
                            #    if x2['begin'] <= x['begin'] <= x2['end'] and x2['status'] == 'dead':
                            #        alive_check = False
                            #        break
                            status_check = True
                            if x['status'] == new_v[-1]['status']:
                                new_v[-1]['end'] = x['end']
                            elif x['end'] - x['begin'] > threshold and x['status'] != 'n/a' and alive_check and status_check:
                                new_v.append(x)
                    new_v[0]['begin'] = 0
                    new_statuses[k] = new_v
                else:
                    if k == 'alive':
                        thresholds = {'alive': 1, 'dead': 1, 'n/a': 1}
                    elif k == 'ult':
                        v = filter_statuses(v, {'has_ult': 0.3})
                        thresholds = {'has_ult': 0.3, 'using_ult': 0.5, 'no_ult': 3, 'n/a': 1}
                        for i, interval in enumerate(v):
                            if interval['status'] == 'n/a':
                                interval['status'] = 'no_ult'
                        v = coalesce_statuses(v)
                    elif k == 'status':
                        thresholds = {'stunned': 0.2, 'asleep': 0.4, 'hacked': 0.2, 'normal': 0.2, 'frozen': 0.2}
                        for i, interval in enumerate(v):
                            if interval['status'] != 'normal' and interval['end'] - interval['begin'] == 0:
                                if i < len(v) - 1 and v[i + 1]['status'] != 'normal':
                                    v[i]['status'] = v[i + 1]['status']
                        v = coalesce_statuses(v)
                    elif k == 'antiheal':
                        thresholds = {'not_antiheal': 1, 'antiheal': 0.4, 'n/a': 1}
                    elif k == 'immortal':
                        thresholds = {'not_immortal': 1, 'immortal': 0.4, 'n/a': 1}
                    elif k == 'nanoboosted':
                        thresholds = {'not_nanoboosted': 1, 'nanoboosted': 0.4, 'n/a': 1}
                    else:
                        print(k)
                        raise Exception('Unknown key')
                    v = filter_statuses(v, thresholds)
                    if k == 'ult':
                        print('FILTERED', v)
                    new_v = []
                    for i, x in enumerate(v):
                        if k == 'ult':
                            if x['status'] != 'no_ult' and x['begin'] < 10:
                                continue
                            if i != 0 and i < len(v) - 1 and x['status'] == 'using_ult' and v[i-1]['status'] == 'no_ult' \
                                    and v[i+1]['status'] == 'has_ult':
                                x['status'] = 'has_ult'
                        if not new_v:
                            new_v.append(x)
                        else:
                            if x['status'] == new_v[-1]['status']:
                                new_v[-1]['end'] = x['end']
                            else:
                                if k == 'status' and x['status'] == 'stunned' and i != len(v) - 1 and i != 0 and \
                                        v[i + 1]['status'] == v[i - 1]['status'] == 'asleep':
                                    continue
                                # if k == 'ult' and x['status'] == 'using_ult' and i != len(v) - 1 and v[i+1]['status'] == 'has_ult':
                                #    continue
                                # if k == 'ult' and x['status'] == 'has_ult' and i != len(v) - 1 and v[i+1]['status'] != 'using_ult':
                                #    continue
                                new_v.append(x)
                    new_v[0]['begin'] = 0
                    new_statuses[k] = new_v

                    print('UPDATED', new_v)
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
            teams[side][ind] = PlayerState(slot, new_statuses, color=colors[side])
        left_team = Team('left', teams['left'])
        left_team.color = self.left_team_color
        for p, v in left_team.player_states.items():
            v.color = self.left_team_color
        right_team = Team('right', teams['right'])
        right_team.color = self.right_team_color
        for p, v in right_team.player_states.items():
            v.color = self.right_team_color
        left_team, right_team = self.annotate_status_effects(left_team, right_team)
        return left_team, right_team

    def annotate_status_effects(self, left_team, right_team):
        thresholds = {'stunned': 0.2, 'asleep': 0.4, 'hacked': 0.2, 'normal': 0.2, 'frozen': 0.2,
                      'discord': 0.2, 'resurrecting': 0.7}
        hurtful_causers = {
            'asleep': ['ana'],
            'stunned': ['brigitte', 'reinhardt', 'roadhog', 'doomfist', 'mccree', 'sigma'],
            'discord': ['zenyatta'],
            'frozen': ['mei'],
            'hacked': ['sombra'],
        }
        indexes = {}
        for k in hurtful_causers.keys():
            indexes[k] = self.model.sets['status'].index(k)
        helpful_causers = {
            'resurrecting': ['mercy']
        }
        for k in helpful_causers.keys():
            indexes[k] = self.model.sets['status'].index(k)
        hurtful_ranges = {'left': {}, 'right': {}}
        for k, hero_set in hurtful_causers.items():
            for side in hurtful_ranges.keys():
                if side == 'left':
                    t = left_team
                else:
                    t = right_team
                hurtful_ranges[side][k] = t.get_hero_range(hero_set)
        helpful_ranges = {'left': {}, 'right': {}}
        for k, hero_set in helpful_causers.items():
            for side in helpful_ranges.keys():
                if side == 'left':
                    t = left_team
                else:
                    t = right_team
                helpful_ranges[side][k] = t.get_hero_range(hero_set)
        statuses = {}
        for s in self.slot_params.keys():
            statuses[s] = []
        for begin_time, status_effect_probs in self.stored_probs.items():

            for s, probs in status_effect_probs.items():
                side = s[0]
                if side == 'left':
                    enemy_side = 'right'
                    t = left_team
                else:
                    enemy_side = 'left'
                    t = right_team

                for effect, effect_index in indexes.items():
                    if effect in hurtful_causers:
                        ranges = hurtful_ranges[enemy_side][effect]
                    else:
                        ranges = helpful_ranges[side][effect]
                    dva_ranges = t.player_states[s[1]].dva_ranges()
                    if not ranges and effect != 'stunned' and not dva_ranges:
                        probs[:, effect_index] = -np.inf
                    else:
                        for t_ind in range(probs.shape[0]):
                            current_time = round(begin_time + (t_ind * self.time_step), 1)
                            if current_time in self.ignored_time_points[s]:
                                continue
                            if effect == 'stunned':
                                if not check_time_in_ranges(current_time, ranges) and not check_time_in_ranges(current_time, dva_ranges):
                                    probs[t_ind, effect_index] = -np.inf
                            elif not check_time_in_ranges(current_time, ranges):
                                probs[t_ind, effect_index] = -np.inf
                v = probs.argmax(axis=1)
                for t_ind in range(v.shape[0]):
                    current_time = round(begin_time + (t_ind * self.time_step), 1)
                    if current_time in self.ignored_time_points[s]:
                        continue
                    label = self.model.sets['status'][v[t_ind]]
                    if statuses[s] and label == statuses[s][-1]['status']:
                        statuses[s][-1]['end'] = current_time
                    else:
                        statuses[s].append(
                            {'begin': current_time, 'end': current_time, 'status': label})
            for s, v in statuses.items():
                for i, interval in enumerate(v):
                    if interval['status'] != 'normal' and interval['end'] - interval['begin'] == 0:
                        if i < len(v) - 1 and v[i + 1]['status'] != 'normal':
                            v[i]['status'] = v[i + 1]['status']
                if s[0] == 'left':
                    team = left_team
                else:
                    team = right_team
                v = coalesce_statuses(v)
                v = filter_statuses(v, thresholds)
                team.player_states[s[1]].statuses['status'] = v
        return left_team, right_team
