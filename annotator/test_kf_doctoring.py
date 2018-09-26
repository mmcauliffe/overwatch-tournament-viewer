import pickle
import time
from collections import defaultdict, Counter
#from annotator.annotate import PlayerState
from annotator.utils import ABILITY_SET

hero_set = ['ana', 'bastion', 'd.va', 'doomfist', 'genji', 'hanzo', 'junkrat', 'lúcio', 'mccree', 'mei', 'mercy', 'moira', 'orisa', 'pharah', 'reaper', 'reinhardt', 'roadhog', 'soldier: 76', 'sombra', 'symmetra', 'torbjörn', 'tracer', 'widowmaker', 'winston', 'zarya', 'zenyatta']

ability_mapping = {'ana': ['biotic grenade', 'sleep dart', ],
                   'bastion': ['configuration: tank',],
                   'd.va': ['boosters', 'call mech', 'micro missiles', 'self-destruct', ],
                   'doomfist': ['meteor strike','rising uppercut', 'rocket punch',  'seismic slam', ],
                   'genji': ['deflect','dragonblade', 'swift strike', ],
                   'hanzo': ['dragonstrike', 'scatter arrow', 'sonic arrow', ],
                   'junkrat': [ 'concussion mine', 'rip-tire',  'steel trap', 'total mayhem', ],
                   'lúcio': ['soundwave', ],
                   'mccree': ['deadeye','flashbang',],
                   'mei': ['blizzard', ],
                   'mercy': ['resurrect', ],
                   'moira': ['biotic orb', 'coalescence', ],
                   'orisa': ['halt!',],
                   'pharah': ['barrage','concussion blast',],
                   'reaper': ['death blossom', ],
                   'reinhardt': ['charge', 'earthshatter','fire strike',],
                   'roadhog': ['chain hook','whole hog', ],
                   'soldier: 76': [ 'helix rockets','tactical visor', ],
                   'sombra': [],
                   'symmetra': ['sentry turret', ],
                   'torbjörn': [ 'forge hammer','turret', ],
                   'tracer': ['pulse bomb', ],
                   'widowmaker': ['venom mine', ],
                   'winston': [ 'jump pack', 'primal rage', ],
                   'zarya': [ 'graviton surge',],
                   'zenyatta': []}
npc_set = ['mech', 'rip-tire', 'shield generator', 'supercharger', 'teleporter', 'turret']
npc_mapping = {'mech': 'd.va', 'rip-tire': 'junkrat', 'shield generator': 'symmetra', 'supercharger': 'orisa', 'teleporter': 'symmetra', 'turret':'torbjörn'}

status_path = r'E:\Data\Overwatch\status_test.pickle'
kf_path = r'E:\Data\Overwatch\kf_test.pickle'

with open(status_path, 'rb') as f:
    states = pickle.load(f)

with open(kf_path, 'rb') as f:
    kf = pickle.load(f)

for p, s in states.items():
    print(p)
    for k, v in s.items():
        print(k)
        print(v)


def close_events(e_one, e_two):
    if e_one['first_hero'] != e_two['first_hero']:
        return False
    if e_one['second_hero'] != e_two['second_hero']:
        return False
    if e_one['first_color'] != e_two['first_color']:
        return False
    if e_one['second_color'] != e_two['second_color']:
        return False
    # if e_one['ability'] != e_two['ability']:
    #    return False
    return True


def merged_event(e_one, e_two):
    if e_one['event'] == e_two['event']:
        return e_one['event']
    elif e_one['duration'] > e_two['duration']:
        return e_one['event']
    return e_two['event']

class Team(object):
    def __init__(self, side, player_states):
        self.side = side
        self.player_states = player_states
        self.color = max(Counter([x.color for x in player_states.values()]).items(), key=lambda x: x[1])[0]

    def heroes_at_time(self, time_point):
        heroes = []
        for k,v in self.player_states.items():
            heroes.append(v.hero_at_time(time_point))
        return heroes

    def has_hero_at_time(self, hero, time_point):
        for k,v in self.player_states.items():
            if v.hero_at_time(time_point) == hero:
                return True
        return False

    def alive_at_time(self, hero, time_point):
        for k,v in self.player_states.items():
            if v.hero_at_time(time_point) == hero:
                return v.alive_at_time(time_point)
        return None

    def get_death_events(self):
        death_events = []
        for k,v in self.player_states.items():
            death_events.extend(v.generate_death_and_revive_events())
        return death_events

class PlayerState(object):
    def __init__(self, player, statuses):
        self.player = player
        self.statuses = statuses
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

    def generate_death_and_revive_events(self):
        deaths = []
        for alive_state in self.statuses['alive']:
            if alive_state['status'] == 'dead':
                deaths.append({'time_point':alive_state['begin'] + 0.2, 'hero': self.hero_at_time(alive_state['begin']), 'color': self.color, 'player': self.player})
        return deaths

    def generate_switches(self):
        switches = []
        for hero_state in self.statuses['hero']:
            switches.append([hero_state['begin'], hero_state['status']])
        return switches

    def generate_ults(self):
        ult_gains, ult_uses = [], []
        for i, ult_state in enumerate(self.statuses['ult']):

            if ult_state['status'] == 'has_ult':
                ult_gains.append([ult_state['begin']])
            elif i > 0 and ult_state['status'] == 'no_ult':
                ult_uses.append([ult_state['begin']])
        return ult_gains, ult_uses

def fix_statuses(lstm_statuses):
    teams = {'left':{}, 'right':{}}
    for k, v in lstm_statuses.items():
        print(k)
        side, ind = k
        new_statuses = {}
        colors = defaultdict(float)
        for interval in v['color']:
            colors[interval['status']] += interval['end'] - interval['begin']
        new_statuses['color'] = max(colors.keys(), key=lambda x: colors[x])

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
                            if s['begin'] <= interval['begin'] < s['end'] and interval['end'] - interval['begin'] < 10:
                                check = True
                                break
                        if check and s['status'] == 'dead':
                            continue
                    new_series.append(interval)
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
    right_team = Team('right', teams['right'])
    return left_team, right_team


def generate_kill_events(kf, left_team, right_team):
    left_color = left_team.color
    right_color = right_team.color
    print('KILL FEED')
    possible_events = []
    for ind, k in enumerate(kf):
        for slot in range(6):
            if slot not in k['slots']:
                continue
            prev_events = []
            if ind != 0:
                if 0 in kf[ind - 1]['slots']:
                    prev_events.append(kf[ind - 1]['slots'][0])
                for j in range(slot, 0, -1):
                    if j in kf[ind - 1]['slots']:
                        prev_events.append(kf[ind - 1]['slots'][j])
            e = k['slots'][slot]
            if 280<=k['time_point']<=281:
                print(e)
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
            if e['second_hero'] in hero_set:
                if not dying_team.has_hero_at_time(e['second_hero'], k['time_point']):
                    continue
                if e['ability'] != 'resurrect' and dying_team.alive_at_time(e['second_hero'],k['time_point']):
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
            if close_events(p['event'], p2['event']) and abs(p2_end - p['time_point']) <= 1.5 and p2['duration'] + p[
                'duration'] < 8:
                better_possible_events[j]['duration'] += p['duration']
                better_possible_events[j]['event'] = merged_event(p, p2)
                break
            elif close_events(p['event'], p2['event']) and p2_end > p['time_point'] > p2['time_point']:
                break

        else:
            better_possible_events.append(p)
    #better_possible_events = [x for x in better_possible_events if x['duration'] > 1]
    for e in better_possible_events:
        #if e['duration'] == 0:
        #    continue
        print(time.strftime('%M:%S', time.gmtime(e['time_point'])), e['time_point'], e['duration'], e['time_point'] + e['duration'], e['event'])
    death_events = sorted(left_team.get_death_events() + right_team.get_death_events(), key= lambda x: x['time_point'])
    actual_events = []
    for de in death_events:
        print(time.strftime('%M:%S', time.gmtime(de['time_point'])), de)
        best_distance = 100
        best_event = None
        for e in better_possible_events:
            if e['event']['second_hero'] != de['hero']:
                continue
            if e['event']['second_color'] != de['color']:
                continue
            dist = abs(e['time_point'] - de['time_point'])
            if dist < best_distance:
                best_event = e
                best_distance = dist
        print(best_event)
        if best_event is None or best_distance > 7:
            continue
        best_event['time_point'] = de['time_point']
        actual_events.append(best_event)
    print('NPC DEATHS')
    for e in better_possible_events:
        if e['event']['second_hero'] in npc_set:
            actual_events.append(e)
            print(time.strftime('%M:%S', time.gmtime(e['time_point'])), e)
    print('REVIVES')
    for e in better_possible_events:
        if e['event']['ability'] == 'resurrect':
            actual_events.append(e)
            print(time.strftime('%M:%S', time.gmtime(e['time_point'])), e)
    return sorted(actual_events, key=lambda x: x['time_point'])

left_team, right_team = fix_statuses(states)

events = generate_kill_events(kf, left_team, right_team)
for e in events:
    print(e)
