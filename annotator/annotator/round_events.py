import os
import json
import cv2
import numpy as np
import re
import keras
import itertools
import time
from collections import defaultdict, Counter

from annotator.config import na_lab, sides, BOX_PARAMETERS
from annotator.game_values import HERO_SET, COLOR_SET, ABILITY_SET, HERO_ONLY_SET
from annotator.utils import get_local_vod, \
    get_local_path,  FileVideoStream, Empty, get_vod_path
from annotator.api_requests import get_annotate_vods_round_events, upload_annotated_round_events

working_dir = r'E:\Data\Overwatch\models'
player_model_dir = os.path.join(working_dir, 'player_status')
player_ocr_model_dir = os.path.join(working_dir, 'player_ocr')
kf_ctc_model_dir = os.path.join(working_dir, 'kill_feed_ctc')
game_model_dir = os.path.join(working_dir, 'game_cnn')
mid_model_dir = os.path.join(working_dir, 'mid')

annotation_dir = r'E:\Data\Overwatch\annotations'
oi_annotation_dir = r'E:\Data\Overwatch\oi_annotations'

time_step = 0.1
frames_per_seq = 100
debug = False

ability_mapping = {'ana': ['biotic grenade', 'sleep dart', ],
                   'bastion': ['configuration: tank', ],
                   'd.va': ['boosters', 'call mech', 'micro missiles', 'self-destruct', ],
                   'doomfist': ['meteor strike', 'rising uppercut', 'rocket punch', 'seismic slam', ],
                   'genji': ['deflect', 'dragonblade', 'swift strike', ],
                   'hanzo': ['dragonstrike', 'scatter arrow', 'sonic arrow', ],
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
                   'soldier: 76': ['helix rockets', 'tactical visor', ],
                   'sombra': [],
                   'symmetra': ['sentry turret', ],
                   'torbjörn': ['forge hammer', 'turret', ],
                   'tracer': ['pulse bomb', ],
                   'widowmaker': ['venom mine', ],
                   'winston': ['jump pack', 'primal rage', ],
                   'zarya': ['graviton surge', ],
                   'zenyatta': []}
npc_set = ['mech', 'rip-tire', 'shield generator', 'supercharger', 'teleporter', 'turret']
npc_mapping = {'mech': 'd.va', 'rip-tire': 'junkrat', 'shield generator': 'symmetra', 'supercharger': 'orisa',
               'teleporter': 'symmetra', 'turret': 'torbjörn'}


def format_time(time_point):
    return time.strftime('%M:%S', time.gmtime(time_point))

def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


class KillFeedAnnotator(object):
    time_step = 0.1

    def __init__(self, model_directory, film_format, spectator_mode, debug=False, half_npc_slots=False):
        self.half_npc_slots = half_npc_slots
        self.model_directory = model_directory
        label_path = os.path.join(self.model_directory, 'labels_set.txt')
        self.label_set = load_set(label_path)
        self.debug = debug
        final_output_weights = os.path.join(self.model_directory, 'kf_weights.h5')
        final_output_json = os.path.join(self.model_directory, 'kf_model.json')
        with open(final_output_json, 'r') as f:
            loaded_model_json = f.read()
        model = keras.models.model_from_json(loaded_model_json)
        model.load_weights(final_output_weights)
        self.model = keras.models.Model(inputs=[model.input[0]],
                                             outputs=[model.get_layer('softmax').output])

        #self.spectator_modes = load_set(os.path.join(self.model_directory, 'spectator_mode_set.txt'))
        #self.spectator_mode_count = len(self.spectator_modes)
        self.kill_feed = []

        self.params = BOX_PARAMETERS[film_format]['KILL_FEED_SLOT']

        self.shape = (6, self.params['WIDTH'], self.params['HEIGHT'], 3)
        #self.spectator_mode_input = self._format_spec_mode_input(6, spectator_mode)

        if self.debug:
            self.caps = {}
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap = cv2.VideoWriter(r'E:\Data\Overwatch\test_full.avi', fourcc, 10.0, (1280, 720))
            for slot in range(6):
                self.caps[slot] = cv2.VideoWriter(r'E:\Data\Overwatch\test_{}.avi'.format(slot), fourcc, 10.0,
                                                (self.params['WIDTH'], self.params['HEIGHT']))

    def process_frame(self, frame, time_point):
        if self.half_npc_slots:
            shift = 0
            cur_kf = {}
            for slot in range(6):
                to_predicts = np.zeros(self.shape, dtype=np.uint8)
                for slot in range(6):
                    x = self.params['X']
                    y = self.params['Y']
                    y += (self.params['HEIGHT'] + self.params['MARGIN']) * (slot)
                    box = frame[y - shift: y + self.params['HEIGHT'] - shift,
                          x: x + self.params['WIDTH']]
                    #cv2.imshow('slot_{}'.format(slot), box)
                    if self.debug:
                        self.caps[slot].write(box)
                    to_predicts[0, ...] = np.swapaxes(box, 1, 0)[None]
                    out = self.model.predict_on_batch([to_predicts])
                    s = self.convert_kf_ctc_output(out[0])
                    if s['second_hero'] == 'n/a' and s['second_color'] == 'n/a':
                        break
                    if s['second_hero'] != 'n/a':
                        cur_kf[slot] = s
                        if s['second_hero'] not in HERO_ONLY_SET + ['b.o.b.']:
                            shift += int(self.params['HEIGHT'] / 4) + 5
            self.kill_feed.append({'time_point': time_point, 'slots': cur_kf})
            #if cur_kf:
            #    print(cur_kf)
            #    cv2.waitKey()

        else:
            to_predicts = np.zeros(self.shape, dtype=np.uint8)
            for slot in range(6):
                x = self.params['X']
                y = self.params['Y']
                y += (self.params['HEIGHT'] + self.params['MARGIN']) * (slot)
                box = frame[y: y + self.params['HEIGHT'],
                      x: x + self.params['WIDTH']]
                if self.debug:
                    self.caps[slot].write(box)
                to_predicts[slot, ...] = np.swapaxes(box, 1, 0)[None]
            self.annotate(to_predicts, time_point)
            if self.debug:
                self.cap.write(frame)

    def annotate(self, to_predicts, time_point):
        cur_kf = {}
        out = self.model.predict_on_batch([to_predicts])
        for slot in range(6):
            s = self.convert_kf_ctc_output(out[slot])
            if s['second_hero'] != 'n/a':
                cur_kf[slot] = s
        self.kill_feed.append({'time_point': time_point, 'slots': cur_kf})

    def cleanup(self):
        if self.debug:
            self.cap.release()
            for slot in range(6):
                self.caps[slot].release()


    def _format_spec_mode_input(self, length, spectator_mode):
        input = np.zeros((length, 62, self.spectator_mode_count))
        m = sparsify(np.array([self.spectator_modes.index(spectator_mode)]), self.spectator_mode_count)
        for i in range(length):
            input[i, :] = m
        return input

    def kf_labels_to_text(self, ls):
        ret = []
        for c in ls:
            #print(c)
            if c >= len(self.label_set):
                continue
            #print(self.label_set[c])
            ret.append(self.label_set[c])
        return ret

    def convert_kf_ctc_output(self, out):
        out_best = list(np.argmax(out[2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        #print(out_best)
        ret = self.kf_labels_to_text(out_best)
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
            elif i in HERO_SET:
                data['second_hero'] = i
        return data

    def generate_death_events(self):
        pass

    def generate_kill_events(self, left_team, right_team):
        from copy import deepcopy
        left_color = left_team.color
        right_color = right_team.color
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
        actual_events = []
        integrity_check = set()
        for de in death_events:
            #print(time.strftime('%M:%S', time.gmtime(de['time_point'])), de)
            best_distance = 100
            best_event = None
            for e in better_possible_events:
                if e['event']['second_hero'] != de['hero']:
                    continue
                if e['event']['second_color'] != de['color']:
                    continue
                dist = abs(e['time_point'] - de['time_point'])
                if dist < best_distance:
                    best_event = deepcopy(e)
                    best_distance = dist
            #print(best_event)
            if best_event is None or best_distance > 7:
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
                integ = (e['time_point'], e['event']['second_hero'], e['event']['second_color'])
                if integ in integrity_check:
                    continue
                actual_events.append(e)
                integrity_check.add(integ)
                #print(time.strftime('%M:%S', time.gmtime(e['time_point'])), e)
        return sorted(actual_events, key=lambda x: x['time_point'])


class MidAnnotator(object):
    resize_factor = 0.5
    time_step = 0.1

    def __init__(self, model_directory, film_format):
        self.film_format = film_format
        self.model_directory = model_directory
        final_output_weights = os.path.join(self.model_directory, 'mid_weights.h5')
        final_output_json = os.path.join(self.model_directory, 'mid_model.json')
        with open(final_output_json, 'r') as f:
            loaded_model_json = f.read()
        self.model = keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(final_output_weights)
        self.params = BOX_PARAMETERS[self.film_format]['MID']
        self.max_num_sequences = 10
        self.shape = (self.max_num_sequences, frames_per_seq, int(self.params['HEIGHT'] * self.resize_factor), int(self.params['WIDTH'] * self.resize_factor), 3)
        self.to_predict = np.zeros(self.shape, dtype=np.uint8)
        self.process_index = 0
        self.num_sequences = 0
        self.begin_time = 0
        mid_set_files = {
            'overtime': os.path.join(self.model_directory, 'overtime_set.txt'),
            'point_status': os.path.join(self.model_directory, 'point_status_set.txt'),
        }

        mid_end_set_files = {
            'attacking_color': os.path.join(self.model_directory, 'attacking_color_set.txt'),
            'map': os.path.join(self.model_directory, 'map_set.txt'),
            'map_mode': os.path.join(self.model_directory, 'map_mode_set.txt'),
            'round_number': os.path.join(self.model_directory, 'round_number_set.txt'),
            'spectator_mode': os.path.join(self.model_directory, 'spectator_mode_set.txt'),
        }

        self.sets = {}

        for k, v in mid_set_files.items():
            self.sets[k] = load_set(v)

        self.end_sets = {}
        for k, v in mid_end_set_files.items():
            self.end_sets[k] = load_set(v)

        self.statuses = {k: [] for k in list(self.sets.keys()) + list(self.end_sets.keys())}

    def process_frame(self, frame):
        box = frame[self.params['Y']: self.params['Y'] + self.params['HEIGHT'],
              self.params['X']: self.params['X'] + self.params['WIDTH']]
        box = cv2.resize(box, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        self.to_predict[self.num_sequences, self.process_index, ...] = box[None]
        self.process_index += 1
        if self.process_index == frames_per_seq:
            self.num_sequences += 1
            self.process_index = 0
            if self.num_sequences == self.max_num_sequences:
                self.annotate()
                self.to_predict = np.zeros(self.shape, dtype=np.uint8)
                self.num_sequences = 0
                self.begin_time += self.max_num_sequences * frames_per_seq * self.time_step
                print(self.begin_time)

    def annotate(self):
        if self.num_sequences == 0:
            return
        lstm_output = self.model.predict([self.to_predict])
        for i in range(self.num_sequences):
            for output_ind, (output_key, s) in enumerate(list(self.sets.items()) + list(self.end_sets.items())):
                if output_key in self.end_sets:
                    label_inds = lstm_output[output_ind].argmax(axis=1)
                    lstm_label = s[label_inds[i]]
                    #print(output_key, lstm_label)
                    if lstm_label == 'n/a' and output_key != 'attacking_color':
                        continue
                    if len(self.statuses[output_key]) == 0:
                        self.statuses[output_key].append(
                            {'begin': self.begin_time + time_step * frames_per_seq * (i), 'end': self.begin_time + time_step * frames_per_seq * (i+1), 'status': lstm_label})
                    else:
                        if lstm_label == self.statuses[output_key][-1]['status']:
                            self.statuses[output_key][-1]['end'] = self.begin_time + time_step * frames_per_seq * (i+1)
                        else:
                            self.statuses[output_key].append(
                                {'begin': self.begin_time + time_step * frames_per_seq * (i), 'end': self.begin_time + time_step * frames_per_seq * (i+1),
                                 'status': lstm_label})
                else:
                    label_inds = lstm_output[output_ind].argmax(axis=2)
                    for t_ind in range(frames_per_seq):
                        current_time = self.begin_time + (t_ind * time_step) +time_step * frames_per_seq *i
                        lstm_label = s[label_inds[i, t_ind]]
                        #print(output_key, lstm_label)
                        if lstm_label != 'n/a':
                            if len(self.statuses[output_key]) == 0:
                                self.statuses[output_key].append({'begin': 0, 'end': 0, 'status': lstm_label})
                            else:
                                if lstm_label == self.statuses[output_key][-1]['status']:
                                    self.statuses[output_key][-1]['end'] = current_time
                                else:
                                    self.statuses[output_key].append(
                                        {'begin': current_time, 'end': current_time, 'status': lstm_label})
            #cv2.imshow('frame', self.to_predict[i][0])
            #cv2.waitKey(0)

    def generate_round_properties(self):
        actual_overtime = []
        for i, r in enumerate(self.statuses['overtime']):
            if r['status'] == 'not_overtime' and r['end'] - r['begin'] < 2:
                continue
            if len(actual_overtime) and r['status'] == actual_overtime[-1]['status']:
                actual_overtime[-1]['end'] = r['end']
            else:
                if len(actual_overtime) and actual_overtime[-1]['end'] != r['begin']:
                    actual_overtime[-1]['end'] = r['begin']
                actual_overtime.append(r)
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

        for k in ['attacking_color', 'map', 'map_mode', 'round_number', 'spectator_mode']:
            counts = defaultdict(float)
            for r in self.statuses[k]:
                counts[r['status']] += r['end'] - r['begin']
            out_props[k] = max(counts, key=lambda x: counts[x])
        out_props['overtime'] = actual_overtime
        out_props['points'] = actual_points
        return out_props


class PlayerAnnotator(object):
    time_step = 0.1

    def __init__(self, model_directory, ocr_directory, film_format, spectator_mode, left_team_color, right_team_color, player_names, debug=False):
        self.model_directory = model_directory
        self.debug = debug
        self.left_team_color = left_team_color
        self.right_team_color = right_team_color
        self.player_names = player_names
        player_set_files = {
            'hero': os.path.join(self.model_directory, 'hero_set.txt'),
            'alive': os.path.join(self.model_directory, 'alive_set.txt'),
            'ult': os.path.join(self.model_directory, 'ult_set.txt'),
             #'antiheal': os.path.join(model_directory, 'antiheal_set.txt'),
             #'asleep': os.path.join(model_directory, 'asleep_set.txt'),
             #'frozen': os.path.join(model_directory, 'frozen_set.txt'),
             #'hacked': os.path.join(model_directory, 'hacked_set.txt'),
             #'stunned': os.path.join(model_directory, 'stunned_set.txt'),
        }
        player_end_set_files = {
        }

        final_output_weights = os.path.join(self.model_directory, 'player_weights.h5')
        final_output_json = os.path.join(self.model_directory, 'player_model.json')
        with open(final_output_json, 'r') as f:
            loaded_model_json = f.read()
        self.model = keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(final_output_weights)

        self.spectator_modes = load_set(os.path.join(self.model_directory, 'spectator_mode_set.txt'))
        self.spectator_mode_count = len(self.spectator_modes)

        self.sides = load_set(os.path.join(self.model_directory, 'side_set.txt'))
        self.side_count = len(self.sides)

        self.colors = load_set(os.path.join(self.model_directory, 'color_set.txt'))
        self.color_count = len(self.colors)
        self.sets = {}
        for k, v in player_set_files.items():
            self.sets[k] = load_set(v)
        self.end_sets = {}
        for k, v in player_end_set_files.items():
            self.end_sets[k] = load_set(v)
        self.statuses = {}
        for side in ['left', 'right']:
            for i in range(6):
                self.statuses[(side, i)] = {k: [] for k in list(self.sets.keys()) + list(self.end_sets.keys())}
        #self.ocr_directory = ocr_directory
        #self.character_set = load_set(os.path.join(self.ocr_directory, 'labels_set.txt'))

        #self.player_names = {}
        #for side in ['left', 'right']:
        #    for i in range(6):
        #        self.player_names[(side, i)] = defaultdict(int)

        #final_output_weights = os.path.join(self.ocr_directory, 'ocr_weights.h5')
        #final_output_json = os.path.join(self.ocr_directory, 'ocr_model.json')
        #with open(final_output_json, 'r') as f:
        #    loaded_model_json = f.read()
        #model = keras.models.model_from_json(loaded_model_json)
        #model.load_weights(final_output_weights)
        #self.ocr_model = keras.models.Model(inputs=[model.input[0], model.input[1]],
        #                                     outputs=[model.get_layer('softmax').output])

        self.left_params = BOX_PARAMETERS[film_format]['LEFT']
        self.right_params = BOX_PARAMETERS[film_format]['RIGHT']
        self.shape = (12, frames_per_seq, self.left_params['HEIGHT'], self.left_params['WIDTH'], 3)
        self.to_predicts = np.zeros(self.shape, dtype=np.uint8)
        self.process_index = 0
        self.begin_time = 0
        self.spectator_mode_input = self._format_spec_mode_input(spectator_mode.lower())
        self.side_input = self._format_side_input()
        self.color_input = self._format_color_input()
        #self.ocr_spectator_mode_input = self._format_ocr_spec_mode_input(spectator_mode)

        if self.debug:
            self.caps = {}
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            for player in self.players:
                self.caps[player] = cv2.VideoWriter(r'E:\Data\Overwatch\test_{}_{}.avi'.format(*player), fourcc, 10.0,
                                                (self.left_params['WIDTH'], self.left_params['HEIGHT']))

    def cleanup(self):
        if self.debug:
            for player in self.players:
                self.caps[player].release()

    def process_frame(self, frame):
        name_boxes = {}
        for i, player in enumerate(self.players):
            side, pos = player
            if side == 'left':
                params = self.left_params
            else:
                params = self.right_params
            x = params['X']
            x += (params['WIDTH'] + params['MARGIN']) * (pos)

            box = frame[params['Y']: params['Y'] + params['HEIGHT'],
                  x: x + params['WIDTH']]
            name_boxes[player] = box[34:46, :]

            self.to_predicts[i, self.process_index, ...] = box[None]
            if self.debug:
                self.caps[player].write(box)

        self.process_index += 1
        #self.annotate_names(name_boxes)
        if self.process_index == frames_per_seq:
            self.annotate_statuses()
            #print(self.statuses)
            self.process_index = 0
            self.begin_time += frames_per_seq * self.time_step
            self.to_predicts = np.zeros(self.shape, dtype=np.uint8)

    @property
    def players(self):
        return list(self.statuses.keys())

    def labels_to_text(self, ls):
        ret = []
        for c in ls:
            if c >= len(self.character_set):
                continue
            ret.append(self.character_set[c])
        return ret

    #def decode_batch(self, word_batch):

    #    out = self.ocr_model.predict_on_batch([word_batch, self.ocr_spectator_mode_input])
    #    ret = []
    #    for j in range(out.shape[0]):
    #        out_best = list(np.argmax(out[j, 2:], 1))
    #        out_best = [k for k, g in itertools.groupby(out_best)]
    #        outstr = self.labels_to_text(out_best)
    #        ret.append(outstr)
    #    return ret

    #def annotate_names(self, name_boxes):
    #    for player, name_box in name_boxes.items():
    #        gray = cv2.cvtColor(name_box, cv2.COLOR_BGR2GRAY)
    #        input = np.expand_dims(cv2.threshold(gray, 0, 255,
    #                                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1], -1)
    #        input = np.swapaxes(input, 0, 1)[None]
    #        label = self.decode_batch(input)[0]
    #        self.player_names[player][''.join(label)] += 1

    def _format_spec_mode_input(self, spectator_mode):
        input = np.zeros((12, 100, self.spectator_mode_count))
        if spectator_mode == 'status':
            spectator_mode = 'overwatch league'
        m = sparsify(np.array([self.spectator_modes.index(spectator_mode)]), self.spectator_mode_count)
        for i in range(12):
            for j in range(100):
                input[i, j, :] = m
        return input

    def _format_ocr_spec_mode_input(self, spectator_mode):
        input = np.zeros((1,16, self.spectator_mode_count))
        m = sparsify(np.array([self.spectator_modes.index(spectator_mode)]), self.spectator_mode_count)
        for i in range(16):
            input[0,i, :] = m
        return input

    def _format_side_input(self):
        input = np.zeros((12, frames_per_seq, self.side_count))
        for i, player in enumerate(self.players):
            if i < 6:
                side = 'left'
            else:
                side = 'right'
            m = sparsify(np.array([self.sides.index(side)]), self.side_count)
            for j in range(frames_per_seq):
                input[i, j, :] = m
        return input

    def _format_color_input(self):
        input = np.zeros((12, frames_per_seq, self.color_count))
        for i, player in enumerate(self.players):
            if i < 6:
                color = self.left_team_color
            else:
                color = self.right_team_color
            m = sparsify(np.array([self.colors.index(color.lower())]), self.color_count)
            for j in range(frames_per_seq):
                input[i, j, :] = m
        return input

    def annotate_statuses(self):
        if self.process_index == 0:
            return
        lstm_output = self.model.predict([self.to_predicts, self.spectator_mode_input, self.side_input]) #, self.color_input])
        #print(len(lstm_output), lstm_output[0].shape)
        for i, player in enumerate(self.statuses.keys()):
            side = player[0]
            for output_ind, (output_key, s) in enumerate(list(self.sets.items()) + list(self.end_sets.items())):
                #print(lstm_output[output_ind].shape)
                if output_key in self.end_sets:
                    label_inds = lstm_output[output_ind][i].argmax(axis=0)
                    lstm_label = s[label_inds]
                    if lstm_label != 'n/a':
                        if len(self.statuses[player][output_key]) == 0:
                            self.statuses[player][output_key].append(
                                {'begin': self.begin_time, 'end': self.begin_time + time_step * frames_per_seq, 'status': lstm_label})
                        else:
                            if lstm_label == self.statuses[player][output_key][-1]['status']:
                                self.statuses[player][output_key][-1]['end'] = self.begin_time + time_step * frames_per_seq
                            else:
                                self.statuses[player][output_key].append(
                                    {'begin': self.begin_time, 'end': self.begin_time + time_step * frames_per_seq,
                                     'status': lstm_label})

                else:
                    label_inds = lstm_output[output_ind][i].argmax(axis=1)
                    #print(label_inds.shape)
                    for t_ind in range(frames_per_seq):
                        current_time = self.begin_time + (t_ind * time_step)
                        #print(label_inds[0, t_ind])
                        #print(output_key, s)
                        #print(s[label_inds[0, t_ind]])
                        lstm_label = s[label_inds[t_ind]]
                        if lstm_label != 'n/a':
                            if len(self.statuses[player][output_key]) == 0:
                                self.statuses[player][output_key].append({'begin': 0, 'end': 0, 'status': lstm_label})
                            else:
                                if lstm_label == self.statuses[player][output_key][-1]['status']:
                                    self.statuses[player][output_key][-1]['end'] = current_time
                                # elif statuses[player][output_key][-1]['end'] - statuses[player][output_key][-1][
                                #    'begin'] < 0.5:
                                #    if len(statuses[player][output_key]) > 2 and lstm_label == \
                                #            statuses[player][output_key][-2]['status']:
                                #        del statuses[player][output_key][-1]
                                #        statuses[player][output_key][-1]['end'] = current_time

                                else:
                                    self.statuses[player][output_key].append(
                                        {'begin': current_time, 'end': current_time, 'status': lstm_label})

    def generate_teams(self):
        teams = {'left': {}, 'right': {}}
        for k, v in self.statuses.items():
            print(k)
            print(v)
            side, ind = k
            new_statuses = {}
            new_statuses['player_name'] = self.player_names[k]
            if side == 'left':
                new_statuses['color'] = self.left_team_color
            else:
                new_statuses['color'] = self.right_team_color

            new_series = []
            for interval in v['alive']:
                if interval['end'] - interval['begin'] > 1:
                    if len(new_series) > 0 and new_series[-1]['status'] == interval['status']:
                        new_series[-1]['end'] = interval['end']
                    else:
                        if len(new_series) > 0 and interval['begin'] != new_series[-1]['end']:
                            interval['begin'] = new_series[-1]['end']
                        new_series.append(interval)

            new_statuses['alive'] = new_series
            new_series = []
            for interval in v['hero']:
                if interval['end'] - interval['begin'] > 5:
                    if len(new_series) > 0 and new_series[-1]['status'] == interval['status']:
                        new_series[-1]['end'] = interval['end']
                    else:
                        if len(new_series) > 0 and interval['begin'] != new_series[-1]['end']:
                            interval['begin'] = new_series[-1]['end']
                        if interval['begin'] != 0:
                            check = False
                            for s in new_statuses['alive']:
                                # print(s)
                                if s['begin'] <= interval['begin'] < s['end'] and \
                                        interval['end'] - interval['begin'] < 10:
                                    check = True
                                    break
                            if check and s['status'] == 'dead':
                                continue
                        new_series.append(interval)
            new_statuses['ult'] = v['ult']
            new_statuses['hero'] = new_series
            death_events = []
            revive_events = []
            for i, interval in enumerate(new_statuses['alive']):
                if interval['status'] == 'dead':
                    death_events.append(interval['begin'])
                    if interval['end'] - interval['begin'] <= 9.8 and i < len(v['alive']) - 1:
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


def sparsify(y, n_classes):
    'Returns labels in binary NumPy array'
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])


def predict_on_video(video_path, begin, end, spectator_mode, film_format, sequences, r):
    print(r)
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
    from datetime import timedelta
    output_video = True
    print('beginning prediction')

    print(begin, end, end - begin)
    kill_feed_annotator = KillFeedAnnotator(kf_ctc_model_dir, film_format, spectator_mode
                                            , debug=debug, half_npc_slots=r['spectator_mode']=='Status')
    player_names = {}
    player_mapping = {}
    for t in r['game']['match']['event']['teams']:
        for p in t['players']:
            player_mapping[p['id']] = p['name']
    for p in r['game']['left_team']['players']:
        player_names[('left', p['player_index'])] = player_mapping[p['player']]
    for p in r['game']['right_team']['players']:
        player_names[('right', p['player_index'])] = player_mapping[p['player']]
    print('PLAYERS', player_names)
    player_annotator = PlayerAnnotator(player_model_dir, player_ocr_model_dir, film_format, spectator_mode,
                                       r['game']['left_team']['color'].lower(), r['game']['right_team']['color'].lower(),
                                       player_names, debug=debug)
    for s in sequences:
        print(s)
        time = s['begin']
        fvs = FileVideoStream(video_path, s['begin'] + begin, s['end'] + begin, time_step, real_begin=begin).start()
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            player_annotator.process_frame(frame)
            kill_feed_annotator.process_frame(frame, time_point)
            print(time_point)

        player_annotator.annotate_statuses()
    kill_feed_annotator.cleanup()
    player_annotator.cleanup()
    # error
    if debug:
        import pickle
        status_path = r'E:\Data\Overwatch\status_test.pickle'
        with open(status_path, 'wb') as f:
            pickle.dump(player_annotator.statuses, f)
        status_path = r'E:\Data\Overwatch\kf_test.pickle'
        with open(status_path, 'wb') as f:
            pickle.dump(kill_feed_annotator.kill_feed, f)
    # print('lstm', )
    death_events = kill_feed_annotator.generate_death_events()
    left_team, right_team = player_annotator.generate_teams()
    left_team.color = r['game']['left_team']['color'].lower()
    right_team.color = r['game']['right_team']['color'].lower()
    kill_feed_events = kill_feed_annotator.generate_kill_events(left_team, right_team)
    print(kill_feed_events)
    # for k, v in player_states.items():
    #    print(k)
    #    print('switches', v.generate_switches())
    #    ug, uu = v.generate_ults()
    #    print('ult_gains', ug)
    #    print('ult_uses', uu)
    data_player_states = {}
    ult_gain_counts = defaultdict(int)
    ult_use_counts = defaultdict(int)
    for t in [left_team, right_team]:
        side = t.side
        for k, v in t.player_states.items():
            k = '{}_{}'.format(side, k)
            data_player_states[k] = {}
            switches = v.generate_switches()
            ug, uu = v.generate_ults(mech_deaths=kill_feed_annotator.mech_deaths)
            data_player_states[k]['player_name'] = v.name
            data_player_states[k]['switches'] = switches
            data_player_states[k]['ult_gains'] = ug
            for u in ug:
                ult_gain_counts[u] += 1
            data_player_states[k]['ult_uses'] = uu
            for u in uu:
                ult_use_counts[u] += 1
    min_time = None
    for time_point, count in list(ult_gain_counts.items()) + list(ult_use_counts.items()):
        if count > 5:
            if min_time is None or time_point < min_time:
                min_time = time_point
    if min_time is not None:
        for t in [left_team, right_team]:
            side = t.side
            for k, v in t.player_states.items():
                k = '{}_{}'.format(side, k)
                data_player_states[k]['ult_gains'] = [x for x in data_player_states[k]['ult_gains'] if x < min_time]
                data_player_states[k]['ult_uses'] = [x for x in data_player_states[k]['ult_uses'] if x < min_time]
    #if debug:
    #    error
    return {'player': data_player_states, 'kill_feed': kill_feed_events, 'left_color': left_team.color,
            'right_color': right_team.color}


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
        self.color = self.statuses['color']

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
        ult_gains, ult_uses = [], []
        end_time = self.statuses['alive'][-1]['end']
        for i, ult_state in enumerate(self.statuses['ult']):
            if i > 0 and ult_state['status'] == 'no_ult' and ult_state['end'] - ult_state['begin'] < 1 and ult_state['end'] != end_time:
                continue
            if self.hero_at_time(ult_state['begin']) == 'd.va':
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
                    ult_gains.append(ult_state['begin'])
                elif i > 0 and ult_state['status'] == 'no_ult':
                    for s in switches:
                        if abs(ult_state['begin'] - s[0]) < 5:
                            break
                    else:
                        ult_uses.append(ult_state['begin'])
        print('ult_gains', ult_gains)
        print('ult_uses', ult_uses)
        return ult_gains, ult_uses



def get_replays(video_path, begin, end, film_format):
    replay_time_step = 5
    debug = False
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'
    print('beginning prediction')
    time_step = 0.1
    frames = (end - begin) / time_step
    num_players = 1
    fvs = FileVideoStream(video_path, begin, end, replay_time_step).start()
    replay_params = BOX_PARAMETERS[film_format]['REPLAY']
    pause_params = BOX_PARAMETERS[film_format]['PAUSE']

    print(begin, end, end - begin)
    frame_ind = 0
    begin_margin = 2.5
    end_margin = 0.6
    replays_found = []
    pauses_found = []
    while True:
        try:
            frame, time_point = fvs.read()
        except Empty:
            break
        time_point = round(time_point, 1)

        print(time_point)
        frame_ind += 1
        box = frame[replay_params['Y']: replay_params['Y'] + replay_params['HEIGHT'],
              replay_params['X']:replay_params['X'] + replay_params['WIDTH']]
        label = pytesseract.image_to_string(box, config=tessdata_dir_config)
        if label.lower() == 'replay':
            replays_found.append(time_point)

        box = frame[pause_params['Y']: pause_params['Y'] + pause_params['HEIGHT'],
              pause_params['X']:pause_params['X'] + pause_params['WIDTH']]

        hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([15, 0, 150])
        upper_orange = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = 255 - mask
        continue
        if np.mean(mask) > 190:  # FIXME

            pauses_found.append(time_point)

    print(replays_found)
    print(pauses_found)
    replays = []
    for t in replays_found:
        already_done = False
        for r in replays:
            if r['end'] >= t >= r['begin']:
                already_done = True
                break
        if already_done:
            continue
        r_b = t - 15 + begin
        r_e = t + 15 + begin
        fvs = FileVideoStream(video_path, r_b, r_e, time_step, real_begin=begin).start()
        replay = {'begin': None}
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            time_point = round(time_point, 1)
            print(time_point)
            frame_ind += 1
            box = frame[replay_params['Y']: replay_params['Y'] + replay_params['HEIGHT'],
                  replay_params['X']:replay_params['X'] + replay_params['WIDTH']]
            label = pytesseract.image_to_string(box, config=tessdata_dir_config)
            if label.lower() == 'replay' and replay['begin'] is None:
                replay['begin'] = time_point
            if label.lower() == 'replay':
                replay['end'] = time_point
        replay['begin'] -= begin_margin
        replay['end'] += end_margin
        replays.append(replay)

    pauses = []
    for t in pauses_found:
        already_done = False
        for r in pauses:
            if r['end'] >= t >= r['begin']:
                already_done = True
                break
        if already_done:
            continue
        r_b = t - 5 + begin
        fvs = FileVideoStream(video_path, r_b, end, time_step, real_begin=begin).start()
        pause = {'begin': None}
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            time_point = round(time_point, 1)
            print(time_point)
            frame_ind += 1

            box = frame[pause_params['Y']: pause_params['Y'] + pause_params['HEIGHT'],
                  pause_params['X']:pause_params['X'] + pause_params['WIDTH']]

            hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)
            lower_orange = np.array([15, 0, 150])
            upper_orange = np.array([35, 255, 255])
            mask = cv2.inRange(hsv, lower_orange, upper_orange)
            mask = 255 - mask
            if np.mean(mask) > 150 and pause['begin'] is None:
                pause['begin'] = time_point
            if np.mean(mask) > 150:
                pause['end'] = time_point
            pause['begin'] -= 0.1
            pause['end'] += 0.1
        pauses.append(pause)

    externals = sorted(replays + pauses, key=lambda x: x['begin'])
    if not externals:
        sequences = [{'begin': begin - begin, 'end': end - begin}]
    else:
        sequences = []
        prev_time = 0
        for r in externals:
            sequences.append({'begin': prev_time, 'end': r['begin']})
            prev_time = r['end']
        sequences.append({'begin': prev_time, 'end': end - begin})
    return replays, pauses, sequences



def get_round_status(vod_path, begin, end, film_format, sequences):
    mid_annotator = MidAnnotator(mid_model_dir, film_format)
    for s in sequences:
        fvs = FileVideoStream(vod_path, s['begin']+begin, s['end']+begin, mid_annotator.time_step, real_begin=begin).start()
        print(s['begin'], s['end'])
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            mid_annotator.process_frame(frame)

        mid_annotator.annotate()
    print('finished!')
    round_props = mid_annotator.generate_round_properties()
    round_props['begin'] = begin
    round_props['end'] = end
    return round_props


def analyze_rounds(vods):
    game_dir = os.path.join(oi_annotation_dir, 'to_check')
    annotation_dir = os.path.join(oi_annotation_dir, 'annotations')
    for v in vods:
        print(v)
        for r in v['rounds']:
            replays, pauses, sequences = get_replays(get_vod_path(v), r['begin'], r['end'], v['film_format'])
            print('r', replays)
            print('p', pauses)
            print('s', sequences)
            #round_props = get_round_status(get_vod_path(v), r['begin'], r['end'], v['film_format'], sequences)

            data = predict_on_video(get_vod_path(v), r['begin'], r['end'], r['spectator_mode'],
                                    v['film_format'], sequences, r)
            data['replays'] = replays
            data['pauses'] = pauses
            data['round'] = r
            print(r)
            print(data)
            left_team = r['game']['left_team']['players']
            right_team = r['game']['right_team']['players']
            for p in left_team:
                index = p['player_index']
                player = p['player']
                slot = 'left_{}'.format(index)
                data['player'][slot]['player'] = player
            for p in right_team:
                index = p['player_index']
                player = p['player']
                slot = 'right_{}'.format(index)
                data['player'][slot]['player'] = player
            data['round'] = r['id']
            upload_annotated_round_events(data)
            error


def vod_main():
    #test()
    #error
    vods = get_annotate_vods_round_events()
    for v in vods:
        local_path = get_vod_path(v)
        if not os.path.exists(local_path):
            get_local_vod(v)

    analyze_rounds(vods)


if __name__ == '__main__':


    print('loaded model')
    vod_main()
