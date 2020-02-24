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
from annotator.models.crnn import SideKillFeedCRNN
from annotator.models.cnn import KillFeedCNN
from annotator.game_values import HERO_SET, COLOR_SET, ABILITY_SET, KILL_FEED_INFO


ability_mapping = KILL_FEED_INFO['ability_mapping']
npc_set = KILL_FEED_INFO['npc_set']
deniable_ults = KILL_FEED_INFO['deniable_ults']
denying_abilities = KILL_FEED_INFO['denying_abilities']
npc_mapping = KILL_FEED_INFO['npc_mapping']


def close_events(e_one, e_two):
    if e_one['second_hero'] != e_two['second_hero']:
        return False
    if e_one['second_side'] != e_two['second_side']:
        return False
    if e_one['second_hero'] in npc_set:
        if e_one['first_hero'] != e_two['first_hero']:
            if e_one['first_hero'] != 'n/a' and e_two['first_hero'] != 'n/a':
                return False
    else:
        if e_one['first_hero'] != e_two['first_hero']:
            return False
        if e_one['first_side'] != e_two['first_side']:
            return False
        # if e_one['ability'] != e_two['ability']:
        #    return False
    return True


def merged_event(e_one, e_two):
    if e_one['duration'] > e_two['duration']:
        merged_e = {k: v for k,v in e_one['event'].items()}
        if merged_e['first_hero'] == 'n/a' and merged_e['first_side'] in ['left', 'right']:
            merged_e['first_hero'] = e_two['event']['first_hero']
    else:
        merged_e = {k: v for k,v in e_two['event'].items()}
        if merged_e['first_hero'] == 'n/a' and merged_e['first_side'] in ['left', 'right']:
            merged_e['first_hero'] = e_one['event']['first_hero']
    return merged_e


def convert_colors(color, left_color, right_color):
    if 'white' in [left_color, right_color]:
        if color != 'white':
            if left_color != 'white':
                return left_color
            else:
                return right_color
        else:
            return color
    else:
        return color


class KillFeedAnnotator(BaseAnnotator):
    time_step = 0.1
    identifier = 'kill_feed'
    box_settings = 'KILL_FEED_SLOT'

    def __init__(self, film_format, model_directory, exists_model_directory, device, left_color, right_color, half_size_npcs=True,
                 spectator_mode='O', debug=False):
        super(KillFeedAnnotator, self).__init__(film_format, device, debug=debug)
        print('=== SETTING UP KILL FEED ANNOTATOR ===')
        self.slots = range(6)
        self.figure_slot_params(film_format)
        self.half_size_npcs = half_size_npcs
        self.spectator_mode = spectator_mode
        self.left_color = left_color
        self.right_color = right_color
        self.model_directory = model_directory
        label_set = load_set(os.path.join(model_directory, 'labels_set.txt'))
        set_paths = {
            'exist': os.path.join(exists_model_directory, 'exist_set.txt'),
            'size': os.path.join(exists_model_directory, 'size_set.txt'),
        }

        sets = {}
        input_sets = {}
        for k, v in set_paths.items():
            sets[k] = load_set(v)
        self.exists_model = KillFeedCNN(sets, input_sets=input_sets)
        spec_dir = os.path.join(exists_model_directory, self.spectator_mode)
        if os.path.exists(spec_dir):
            exists_model_path = os.path.join(spec_dir, 'model.pth')
            print('Using {} exists model!'.format(self.spectator_mode))
        else:
            exists_model_path = os.path.join(exists_model_directory, 'model.pth')
            print('Using base exists model!')
        self.exists_model.load_state_dict(torch.load(exists_model_path))
        self.exists_model.eval()
        for p in self.exists_model.parameters():
            p.requires_grad = False
        self.exists_model.to(device)

        self.model = SideKillFeedCRNN(label_set)
        spec_dir = os.path.join(model_directory, self.spectator_mode)
        if os.path.exists(spec_dir):
            slot_model_path = os.path.join(spec_dir, 'model.pth')
            print('Using {} kf slot model!'.format(self.spectator_mode))
        else:
            slot_model_path = os.path.join(model_directory, 'model.pth')
            print('Using base kf slot model!')
        self.model.load_state_dict(torch.load(slot_model_path))
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
        self.left_colors = Variable(torch.FloatTensor(self.batch_size, 3).to(device))
        self.right_colors = Variable(torch.FloatTensor(self.batch_size, 3).to(device))
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
        show = self.debug
        left_colors = []
        right_colors = []
        first_empty = False
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
            image = ((image / 255) - 0.5) / 0.5
            #print('load exist image', time.time()-b)
            #b = time.time()
            with torch.no_grad():
                predicteds = self.exists_model({'image': image})
            #print('exist predict', time.time()-b)
            d = {}
            for k, se in self.exists_model.sets.items():
                _, predicteds[k] = torch.max(predicteds[k], 1)
                d[k] = se[predicteds[k][0]]
            if show:
                print(s, 'exists', d)
            if d['exist'] == 'empty' and s == 0:
                continue
            elif d['exist'] == 'empty':
                break
            if images is None:
                images = image
            else:
                images = torch.cat((images, image), 0)
            left_colors.append(self.left_color)
            right_colors.append(self.right_color)
            if d['size'] == 'half':
                #show = True
                shift += int(self.params['HEIGHT']/2)
                #print(shift)

        if images is None:
            return
        #b = time.time()
        left_colors = (((torch.FloatTensor(left_colors) / 255) - 0.5) / 0.5).to(self.device)
        right_colors = (((torch.FloatTensor(right_colors) / 255) - 0.5) / 0.5).to(self.device)
        loadData(self.images, images)
        loadData(self.left_colors, left_colors)
        loadData(self.right_colors, right_colors)

        batch_size = self.images.size(0)
        #print('load kf ctc image', time.time()-b, batch_size)
        #b = time.time()
        with torch.no_grad():
            cur_kf = self.model.parse_image(self.images, self.left_colors, self.right_colors)
        if cur_kf and show:
            print(cur_kf)
            cv2.waitKey()
        if cur_kf:
            self.kill_feed.append({'time_point': time_point, 'slots': cur_kf})
        #print('kf ctc decode', time.time()-b)
        #error
        #print('Frame kill feed generation took: ', time.time()-begin)

    def generate_kill_events(self, left_team, right_team, round_end):
        from copy import deepcopy
        print('GENERATING KILL FEED')
        possible_events = []
        #print(self.kill_feed)

        for ind, k in enumerate(self.kill_feed):
            for e in k['slots']:

                if e['first_hero'] == 'n/a' and len(e['assisting_heroes']) > 0:
                    e['first_hero'] = e['assisting_heroes'][0].replace('_assist', '')
                    e['assisting_heroes'] = e['assisting_heroes'][1:]
                if e['first_hero'] != 'n/a':
                    if e['ability'] == 'resurrect' and e['first_side'] != e['second_side']:  # Trust the first color
                        e['second_side'] = e['first_side']
                    #if e['first_color'] not in [left_color, right_color]:
                    #    continue
                    if e['first_side'] == 'left':
                        killing_team = left_team
                    elif e['first_side'] == 'right':
                        killing_team = right_team
                    else:
                        continue
                    if not killing_team.has_hero_at_time(e['first_hero'], k['time_point']):
                        continue
                    if e['ability'] not in ['primary', 'melee']:
                        if e['ability'] not in ability_mapping[e['first_hero']]:
                            if e['first_hero'] != 'genji':
                                continue
                            else:
                                e['ability'] = 'deflect'
                else:
                    if e['ability'] != 'primary':
                        continue
                if e['second_side'] == 'left':
                    dying_team = left_team
                elif e['second_side'] == 'right':
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
                add_new = True
                for p_ind, poss_e in enumerate(possible_events):
                    if e == poss_e['event'] and poss_e['time_point'] + poss_e['duration'] + 0.5 >= k['time_point']:
                        possible_events[p_ind]['duration'] = k['time_point'] - poss_e['time_point']
                        add_new = False
                        break
                if add_new:
                    possible_events.append({'time_point': k['time_point'], 'duration': 0, 'event': e})
        print('POSSIBLE', possible_events)
        better_possible_events = []
        for i, p in enumerate(possible_events):
            for j, p2 in enumerate(better_possible_events):
                p2_end = p2['time_point'] + p2['duration']
                if close_events(p['event'], p2['event']) and abs(p2_end - p['time_point']) <= 3 and \
                        p2['duration'] + p['duration'] < 8:
                    better_possible_events[j]['event'] = merged_event(p, p2)
                    better_possible_events[j]['duration'] += p['duration']
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
        print('POSSIBLE EVENTS', better_possible_events)
        death_events = sorted(left_team.get_death_events() + right_team.get_death_events(),
                              key=lambda x: x['time_point'])
        print('DEATH EVENTS', death_events)
        actual_events = []
        integrity_check = set()
        for de in death_events:
            #print(time.strftime('%M:%S', time.gmtime(de['time_point'])), de)
            best_distance = 100
            best_event = None
            print()
            print(de)
            if de['time_point'] > round_end:
                continue
            for e in better_possible_events:
                #print(e)
                if e['duration'] < 0.5 and round_end - (e['time_point'] + e['duration']) > 1:
                    continue
                if e['event']['second_hero'] != de['hero']:
                    continue
                if e['event']['second_side'] != de['side']:
                    continue
                if e['event']['first_side'] == de['side']:
                    continue
                dist = abs(e['time_point'] - de['time_point'])
                if dist < best_distance:
                    best_event = deepcopy(e)
                    best_distance = dist
                #print(best_event, best_distance)
            #print(best_event)
            if best_event is None or best_distance > 7:
                if de['hero'] is not None:
                    actual_events.append({'time_point': de['time_point'],
                                          'event': {'first_hero': 'n/a',
                                                    'first_side': 'n/a',
                                                    'ability': 'n/a',
                                                    'second_side': de['side'],
                                                    'second_hero': de['hero']}})
                continue
            if de['time_point'] < best_event['time_point']:
                best_event['time_point'] = de['time_point']
            integ = (best_event['time_point'], best_event['event']['second_hero'], best_event['event']['second_side'])
            print(best_event)
            print(integ, integ in integrity_check, integrity_check)
            if integ in integrity_check:
                continue
            actual_events.append(best_event)
            integrity_check.add(integ)
        npc_events = []
        for e in better_possible_events:
            if e['event']['second_hero'] in npc_set or \
                    (e['event']['second_hero'] in deniable_ults and e['event']['ability'] in denying_abilities):
                for ne in npc_events:
                    if ne['duration'] < 0.6:
                        continue
                    if close_events(ne['event'], e['event']):
                        if e['time_point'] + e['duration'] < ne['time_point'] + 7.3 and ne['duration'] < 7.5:
                            ne['duration'] = e['time_point'] + e['duration'] - ne['time_point']
                            break
                else:
                    integ = (e['time_point'], e['event']['second_hero'], e['event']['second_side'])
                    if integ in integrity_check:
                        continue
                    npc_events.append(e)
                    integrity_check.add(integ)
        npc_events = [x for x in npc_events if x['duration'] > 2]
        print('NPC DEATHS')
        print(npc_events)
        self.mech_deaths = []
        for e in npc_events:
            actual_events.append(e)
            if e['event']['second_hero'] == 'mech':
                self.mech_deaths.append({'time_point': e['time_point'], 'side': e['event']['second_side']})
            #print(time.strftime('%M:%S', time.gmtime(e['time_point'])), e)
        print('REVIVES')
        self.dva_revive_events = []
        for i, e in enumerate(better_possible_events):
            if e['event']['ability'] == 'resurrect' and e['duration'] > 0:
                print(e)
                integ = (round(e['time_point'], 1), e['event']['second_hero'], e['event']['second_side'])
                print(integ, integ in integrity_check)
                if integ in integrity_check:
                    continue
                actual_events.append(e)
                if e['event']['second_hero'] == 'd.va':
                    self.dva_revive_events.append({'time_point': e['time_point'], 'side': e['event']['second_side']})

                integrity_check.add(integ)
                #print(time.strftime('%M:%S', time.gmtime(e['time_point'])), e)

        return sorted(actual_events, key=lambda x: x['time_point'])
