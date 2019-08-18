import os
import torch
import numpy as np
import time
import cv2
from annotator.annotator.classes.base import BaseAnnotator
from torch.autograd import Variable

from annotator.models.cnn import StatusCNN
from annotator.training.helper import load_set
from annotator.training.ctc_helper import loadData
from collections import defaultdict, Counter
from annotator.config import sides, BOX_PARAMETERS
from annotator.models.crnn import KillFeedCRNN
from annotator.models.cnn import KillFeedCNN
from annotator.game_values import HERO_SET, COLOR_SET, ABILITY_SET


ability_mapping = {'ana': ['biotic grenade', 'sleep dart', ],
                   'ashe': ['b.o.b.', 'coach gun', 'dynamite'],
                   'brigitte': ['whip shot', 'shield bash'],
                   'baptiste': [],
                   'bastion': ['configuration: tank', ],
                   'd.va': ['boosters', 'call mech', 'micro missiles', 'self-destruct', ],
                   'doomfist': ['meteor strike', 'rising uppercut', 'rocket punch', 'seismic slam', ],
                   'genji': ['deflect', 'dragonblade', 'swift strike', ],
                   'hanzo': ['dragonstrike', 'scatter arrow', 'sonic arrow', 'storm arrow'],
                   'junkrat': ['concussion mine', 'rip-tire', 'steel trap', 'total mayhem', ],
                   'lúcio': ['soundwave', ],
                   'mccree': ['deadeye', 'flashbang', ],
                   'mei': ['blizzard', ],
                   'mercy': ['resurrect', ],
                   'moira': ['biotic orb', 'coalescence', ],
                   'orisa': ['halt!', ],
                   'pharah': ['barrage', 'concussion blast', ],
                   'reaper': ['death blossom', ],
                   'reinhardt': ['charge', 'earthshatter', 'fire strike', ],
                   'roadhog': ['chain hook', 'whole hog', ],
                   'sigma': ['accretion', 'gravitic flux'],
                   'soldier: 76': ['helix rockets', 'tactical visor', ],
                   'sombra': [],
                   'symmetra': ['sentry turret', ],
                   'torbjörn': ['forge hammer', 'turret', ],
                   'tracer': ['pulse bomb', ],
                   'widowmaker': ['venom mine', ],
                   'winston': ['jump pack', 'primal rage', ],
                   'wrecking ball': ['piledriver', 'grappling claw', 'minefield'],
                   'zarya': ['graviton surge', ],
                   'zenyatta': []}
npc_set = ['mech', 'rip-tire', 'shield generator', 'supercharger', 'teleporter', 'turret', 'immortality field']
npc_mapping = {'mech': 'd.va', 'rip-tire': 'junkrat', 'shield generator': 'symmetra', 'supercharger': 'orisa',
               'teleporter': 'symmetra', 'turret': 'torbjörn', 'immortality field': 'baptiste'}


def close_events(e_one, e_two):
    if e_one['second_hero'] != e_two['second_hero']:
        return False
    if e_one['second_color'] != e_two['second_color']:
        return False
    if e_one['second_hero'] in npc_set:
        if e_one['first_hero'] != e_two['first_hero']:
            if e_one['first_hero'] != 'n/a' and e_two['first_hero'] != 'n/a':
                return False
    else:
        if e_one['first_hero'] != e_two['first_hero']:
            return False
        if e_one['first_color'] != e_two['first_color']:
            return False
        # if e_one['ability'] != e_two['ability']:
        #    return False
    return True


def merged_event(e_one, e_two):
    if e_one['event'] == e_two['event']:
        return e_one['event']
    elif e_one['duration'] > e_two['duration']:
        return e_one['event']
    if e_one['event']['first_hero'] != 'n/a' and e_two['event']['first_hero'] == 'n/a':
        return e_one['event']
    return e_two['event']


class KillFeedAnnotator(BaseAnnotator):
    time_step = 0.1
    identifier = 'kill_feed'
    box_settings = 'KILL_FEED_SLOT'

    def __init__(self, film_format, model_directory, exists_model_directory, device, half_size_npcs=True, spectator_mode='O'):
        super(KillFeedAnnotator, self).__init__(film_format, device)
        self.slots = range(6)
        self.figure_slot_params(film_format)
        self.half_size_npcs = half_size_npcs
        self.spectator_mode = spectator_mode
        self.model_directory = model_directory
        label_set = load_set(os.path.join(model_directory, 'labels_set.txt'))
        spectator_mode_set = load_set(os.path.join(model_directory, 'spectator_mode_set.txt'))
        set_paths = {
            'exist': os.path.join(exists_model_directory, 'exist_label_set.txt'),
        }

        sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)
        self.exists_model = KillFeedCNN(sets)
        self.exists_model.load_state_dict(torch.load(os.path.join(exists_model_directory, 'model.pth')))
        self.exists_model.eval()
        for p in self.exists_model.parameters():
            p.requires_grad = False
        self.exists_model.to(device)

        self.model = KillFeedCRNN(label_set, spectator_mode_set)
        self.model.load_state_dict(torch.load(os.path.join(model_directory, 'model.pth')))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)
        self.to_predict = {}
        self.kill_feed = []
        self.shape = (self.batch_size, 3, int(self.params['HEIGHT'] * self.resize_factor),
                      int(self.params['WIDTH'] * self.resize_factor))
        self.images = Variable(torch.FloatTensor(self.batch_size, 3, int(self.params['HEIGHT'] * self.resize_factor),
                      int(self.params['WIDTH'] * self.resize_factor)).to(device))
        self.spectator_modes = Variable(torch.LongTensor(self.batch_size).to(device))
        for s in self.slot_params.keys():
            self.to_predict[s] = np.zeros(self.shape, dtype=np.uint8)

    def figure_slot_params(self, film_format):
        self.slot_params = {}
        params = BOX_PARAMETERS[film_format]['KILL_FEED_SLOT']
        for s in self.slots:
            self.slot_params[s] = {}
            self.slot_params[s]['x'] = params['X']
            self.slot_params[s]['y'] = params['Y'] + (params['HEIGHT'] + params['MARGIN']) * (s)

    def process_frame(self, frame, time_point):
        #begin = time.time()
        #cv2.imshow('frame', frame)
        shift = 0
        cur_kf = {}
        images = None
        show = False
        spectator_modes = []
        for s, params in self.slot_params.items():

            x = params['x']
            y = params['y']
            box = frame[y - shift: y + self.params['HEIGHT'] - shift,
                  x: x + self.params['WIDTH']]
            if show:
                #cv2.imshow('bw_{}'.format(s), bw)
                cv2.imshow('slot_{}'.format(s), box)
                #cv2.waitKey()
            #b = time.time()
            image = torch.from_numpy(np.transpose(box, axes=(2, 0, 1))[None]).float().to(self.device)
            #print('load exist image', time.time()-b)
            #b = time.time()
            predicteds = self.exists_model({'image': image})
            #print('exist predict', time.time()-b)

            _, predicteds['exist'] = torch.max(predicteds['exist'], 1)
            label = self.exists_model.sets['exist'][predicteds['exist'][0]]
            if label == 'empty':
                break
            if images is None:
                images = image
            else:
                images = torch.cat((images, image), 0)
            spectator_modes.append(self.model.spectator_mode_set.index(self.spectator_mode))
            if label == 'half_sized':
                #show = True
                shift += int(self.params['HEIGHT']/2)
                #print(shift)

        if images is None:
            return
        #b = time.time()
        spectator_modes = torch.LongTensor(spectator_modes).to(self.device)
        loadData(self.images, images)
        loadData(self.spectator_modes, spectator_modes)

        batch_size = self.images.size(0)
        #print('load kf ctc image', time.time()-b, batch_size)
        #b = time.time()
        preds = self.model(self.images, self.spectator_modes)
        #print('kf ctc predict', time.time()-b)
        #b = time.time()
        pred_size = preds.size(0)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds = preds.to('cpu')
        for s in range(batch_size):
            start_ind = s * pred_size
            end_ind = (s + 1) * pred_size
            d = [self.model.label_set[x - 1] for x in preds[start_ind:end_ind] if x != 0]
            #print(d)
            cur_kf[s] = self.convert_kf_ctc_output(d)
        if cur_kf and show:
            print(cur_kf)
            cv2.waitKey()
        if cur_kf:
            self.kill_feed.append({'time_point': time_point, 'slots': cur_kf})
        #print('kf ctc decode', time.time()-b)
        #error
        #print('Frame kill feed generation took: ', time.time()-begin)

    def convert_kf_ctc_output(self, ret):
        #print(ret)
        data = {'first_hero': 'n/a',
                'first_color': 'n/a',
                'assists': [],
                'ability': 'n/a',
                'headshot': 'n/a',
                'second_hero': 'n/a',
                'second_color': 'n/a'}
        first_intervals = []
        second_intervals = []
        ability_intervals = []
        for i in ret:
            if i in ABILITY_SET:
                ability_intervals.append(i)
            if not len(ability_intervals):
                first_intervals.append(i)
            elif i not in ABILITY_SET:
                second_intervals.append(i)

        for i in first_intervals:
            if i in COLOR_SET:
                data['first_color'] = i
            elif i in HERO_SET:
                data['first_hero'] = i
            else:
                if i not in data['assists'] and i.replace('_assist', '') != data['first_hero']:
                    data['assists'].append(i)
        for i in ability_intervals:
            if i.endswith('headshot'):
                data['headshot'] = True
                data['ability'] = i.replace(' headshot', '')
            else:
                data['ability'] = i
                data['headshot'] = False
        for i in second_intervals:
            if i in COLOR_SET:
                data['second_color'] = i
            elif i.endswith('_npc'):
                data['second_hero'] = i.replace('_npc', '')
            elif i in HERO_SET:
                data['second_hero'] = i
        if data['first_hero'] != 'n/a':
            if data['ability'] not in ['primary', 'melee']:
                if data['ability'] not in ability_mapping[data['first_hero']]:
                    data['ability'] = 'primary'
        return data

    def generate_kill_events(self, left_team, right_team):
        from copy import deepcopy
        left_color = left_team.color
        right_color = right_team.color
        left_team_white = False
        if left_color == 'white':
            left_team_white = True
        right_team_white = False
        if right_color == 'white':
            right_team_white = True
        print('KILL FEED')
        possible_events = []
        #print(self.kill_feed)

        for ind, k in enumerate(self.kill_feed):
            for slot in range(6):
                if slot not in k['slots']:
                    continue
                prev_events = []
                if ind != 0:
                    if 0 in self.kill_feed[ind - 1]['slots']:
                        prev_events.append(self.kill_feed[ind - 1]['slots'][0])
                    for j in range(slot, 0, -1):
                        if j in self.kill_feed[ind - 1]['slots']:
                            prev_events.append(self.kill_feed[ind - 1]['slots'][j])
                e = k['slots'][slot]
                #if 280 <= k['time_point'] <= 281:
                #    print(e)
                if e['first_hero'] != 'n/a':
                    if e['first_color'] not in [left_color, right_color]:
                        continue
                    if e['first_color'] == left_color:
                        killing_team = left_team
                    elif e['first_color'] == right_color:
                        killing_team = right_team
                    if not killing_team.has_hero_at_time(e['first_hero'], k['time_point']):
                        continue
                    if e['ability'] != 'primary':
                        if e['ability'] not in ability_mapping[e['first_hero']]:
                            if e['first_hero'] != 'genji':
                                continue
                            else:
                                e['ability'] = 'deflect'
                else:
                    if e['ability'] != 'primary':
                        continue
                if e['second_color'] not in [left_color, right_color]:
                    continue
                if e['second_color'] == left_color:
                    dying_team = left_team
                elif e['second_color'] == right_color:
                    dying_team = right_team

                # check if it's a hero or npc
                if e['second_hero'] in HERO_SET:
                    if not dying_team.has_hero_at_time(e['second_hero'], k['time_point']):
                        continue
                    if e['ability'] != 'resurrect' and dying_team.alive_at_time(e['second_hero'], k['time_point']):
                        continue
                elif e['second_hero'] in npc_set:
                    hero = npc_mapping[e['second_hero']]
                    if not dying_team.has_hero_at_time(hero, k['time_point']):
                        continue
                if e in prev_events:
                    for p_ind, poss_e in enumerate(possible_events):
                        if e == poss_e['event'] and poss_e['time_point'] + poss_e['duration'] + 0.5 >= k['time_point']:
                            possible_events[p_ind]['duration'] = k['time_point'] - poss_e['time_point']
                            break
                else:
                    possible_events.append({'time_point': k['time_point'], 'duration': 0, 'event': e})
        print('POSSIBLE EVENTS', possible_events)
        better_possible_events = []
        for i, p in enumerate(possible_events):
            for j, p2 in enumerate(better_possible_events):
                p2_end = p2['time_point'] + p2['duration']
                if close_events(p['event'], p2['event']) and abs(p2_end - p['time_point']) <= 1.5 and p2['duration'] + \
                        p[
                            'duration'] < 8:
                    better_possible_events[j]['duration'] += p['duration']
                    better_possible_events[j]['event'] = merged_event(p, p2)
                    break
                elif close_events(p['event'], p2['event']) and p2_end > p['time_point'] > p2['time_point']:
                    break

            else:
                if better_possible_events and p == better_possible_events[-1]:
                    continue
                better_possible_events.append(p)
        # better_possible_events = [x for x in better_possible_events if x['duration'] > 1]
        #for e in better_possible_events:
            # if e['duration'] == 0:
            #    continue
            #print(time.strftime('%M:%S', time.gmtime(e['time_point'])), e['time_point'], e['duration'],
            #      e['time_point'] + e['duration'], e['event'])
        death_events = sorted(left_team.get_death_events() + right_team.get_death_events(),
                              key=lambda x: x['time_point'])
        print('DEATH EVENTS', death_events)
        actual_events = []
        integrity_check = set()
        for de in death_events:
            #print(time.strftime('%M:%S', time.gmtime(de['time_point'])), de)
            best_distance = 100
            best_event = None
            print(de)
            for e in better_possible_events:
                print(e)
                if e['event']['second_hero'] != de['hero']:
                    continue
                if e['event']['second_color'] != de['color']:
                    continue
                if e['event']['first_color'] == de['color']:
                    continue
                dist = abs(e['time_point'] - de['time_point'])
                if dist < best_distance:
                    best_event = deepcopy(e)
                    best_distance = dist
                print(best_event, best_distance)
            #print(best_event)
            if best_event is None or best_distance > 7:
                if de['hero'] is not None:
                    actual_events.append({'time_point': de['time_point'],
                                          'event': {'first_hero': 'n/a',
                                                    'ability': 'n/a',
                                                    'second_color': de['color'],
                                                    'second_hero': de['hero']}})
                continue
            best_event['time_point'] = de['time_point']
            integ = (best_event['time_point'], best_event['event']['second_hero'], best_event['event']['second_color'])
            print()
            print(de)
            print(best_event)
            print(integ, integ in integrity_check, integrity_check)
            if integ in integrity_check:
                continue
            actual_events.append(best_event)
            integrity_check.add(integ)
            print(actual_events)
        npc_events = []
        for e in better_possible_events:
            if e['event']['second_hero'] in npc_set:
                for ne in npc_events:
                    if close_events(ne['event'], e['event']):
                        if e['time_point'] + e['duration'] < ne['time_point'] + 7.3 and ne['duration'] < 7.5:
                            ne['duration'] = e['time_point'] + e['duration'] - ne['time_point']
                            break
                else:
                    integ = (e['time_point'], e['event']['second_hero'], e['event']['second_color'])
                    if integ in integrity_check:
                        continue
                    npc_events.append(e)
                    integrity_check.add(integ)
        npc_events = [x for x in npc_events if x['duration'] > 3]
        print('NPC DEATHS')
        self.mech_deaths = []
        for e in npc_events:
            actual_events.append(e)
            if e['event']['second_hero'] == 'mech':
                self.mech_deaths.append({'time_point': e['time_point'], 'color': e['event']['second_color']})
            #print(time.strftime('%M:%S', time.gmtime(e['time_point'])), e)
        print('REVIVES')
        for i, e in enumerate(better_possible_events):
            if e['event']['ability'] == 'resurrect' and e['duration'] > 0:
                print(e)
                integ = (round(e['time_point'], 1), e['event']['second_hero'], e['event']['second_color'])
                print(integ, integ in integrity_check)
                if integ in integrity_check:
                    continue
                actual_events.append(e)
                integrity_check.add(integ)
                #print(time.strftime('%M:%S', time.gmtime(e['time_point'])), e)
        return sorted(actual_events, key=lambda x: x['time_point'])
