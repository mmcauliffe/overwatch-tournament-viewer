import sys
import os
import json

base_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, base_dir)

from .settings import *


class Team(object):
    def __init__(self, name, id, color=None):
        self.name = name
        self.id = id
        self.color = color
        if self.color is None:
            self.color = ''
        self.players = []

    def to_json(self):
        return {'name': self.name,
                'id': self.id}

    @staticmethod
    def from_json(data):
        return Team(data['name'], data['id'])

    def save(self):
        team_dir = os.path.join(annotations_dir, 'teams')
        os.makedirs(team_dir, exist_ok=True)
        path = os.path.join(team_dir, '{}.data'.format(self.id))
        with open(path, 'w', encoding='utf8') as f:
            json.dump(self.to_json(), f, indent=4)


class NPC(object):
    def __init__(self, name):
        self.name = name
        self.deaths = []

    def add_death(self, time_point):
        if time_point not in self.deaths:
            self.deaths.append(time_point)

    def to_json(self):
        return {'deaths': self.deaths}


class Player(object):
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def to_json(self):
        return {'name': self.name,
                'id': self.id}

    @staticmethod
    def from_json(data):
        return Team(data['name'], data['id'])

    def save(self):
        player_dir = os.path.join(annotations_dir, 'players')
        os.makedirs(player_dir, exist_ok=True)
        path = os.path.join(player_dir, '{}.data'.format(self.id))
        with open(path, 'w', encoding='utf8') as f:
            json.dump(self.to_json(), f, indent=4)


# class Event(object):
#    def __init__(self, time_point):
#        self.time_point = time_point

# class DeathEvent(Event):
#    pass

# class UltGainEvent(Event):
#    pass

# class UltUseEvent(Event):
#    pass

# class SwitchEvent(Event):
#    def __init__(self, time_point, new_hero):
#        super(SwitchEvent, self).__init__(time_point)
#        self.new_hero = new_hero

class PlayerPerformance(object):
    def __init__(self, player):
        self.player = player
        self.switches = []
        self.ult_gains = []
        self.ult_uses = []
        self.kills = []
        self.revives = []
        self.deaths = []

    def add_death(self, time_point):
        if time_point not in self.deaths:
            self.deaths.append(time_point)

    @staticmethod
    def from_json(data):
        p = PlayerPerformance(load_player(data['player_id']))
        p.switches = data['switches']
        p.ult_gains = data['ult_gains']
        p.ult_uses = data['ult_uses']
        p.kills = data['kills']
        p.revives = data['revives']
        p.deaths = data['deaths']
        return p

    def to_json(self):
        return {'player_id': self.player.id,
                'switches': self.switches,
                'ult_gains': self.ult_gains,
                'ult_uses': self.ult_uses,
                'kills': self.kills,
                'revives': self.revives,
                'deaths': self.deaths}

    def save(self, round_dir):
        path = os.path.join(round_dir, '{}.data'.format(self.player.id))
        with open(path, 'w', encoding='utf8') as f:
            json.dump(self.to_json(), f)

    def hero_at_time(self, time):
        hero = ''
        for s in self.switches:
            if s[0] > time:
                return hero
            hero = s[1]
        return hero

    def has_ult_at_time(self, time):
        if not self.ult_gains:
            return False
        last_gain = 0
        for ug in self.ult_gains:
            if ug > time:
                break
            last_gain = ug
        if last_gain == 0:
            return False
        last_use = 0
        for uu in self.ult_uses:
            if uu > time:
                break
            last_use = uu
        if last_use >= last_gain:
            return False
        last_switch = 0
        for s in self.switches:
            if s[0] > time:
                break
            last_switch = s[0]
        if last_switch > last_gain:
            return False
        return True

    def add_switch(self, time_point, hero):
        self.switches = [x for x in self.switches if x[0] != time_point]
        self.switches.append((time_point, hero))

    def add_ult_gain(self, time_point):
        if time_point not in self.ult_gains:
            self.ult_gains.append(time_point)

    def add_ult_use(self, time_point):
        if time_point not in self.ult_uses:
            self.ult_uses.append(time_point)

    def add_kill(self, time_point, killed_id, method):
        data_point = (time_point, killed_id, method)
        for i, k in enumerate(self.kills):
            if k[0] == time_point and k[1] == killed_id:
                self.kills[i] = data_point
                break
        else:
            self.kills.append(data_point)


class Game(object):
    def __init__(self, match, id, map, left_team, right_team, begin=None, end=None):
        self.match = match
        self.id = id
        self.left_team = left_team
        self.right_team = right_team
        self.begin = begin
        self.end = end
        self.map = map
        self.rounds = []
        self.style = 'regular'

    def to_json(self):
        return {'match': self.match, 'id': self.id, 'map': self.map,
                'left_team_id': self.left_team.id,
                'left_team_players': [x.id for x in self.left_team.players],
                'right_team_id': self.right_team.id,
                'right_team_players': [x.id for x in self.right_team.players],
                'style': self.style}

    @staticmethod
    def from_json(data):
        left_team = load_team(data['left_team_id'])
        left_team.players = [load_player(x) for x in data['left_team_players']]
        right_team = load_team(data['right_team_id'])
        right_team.players = [load_player(x) for x in data['right_team_players']]
        g = Game(data['match'], data['id'], data['map'], left_team, right_team)
        g.style = data.get('style', 'regular')
        return g

    def save(self):
        game_dir = os.path.join(annotations_dir, self.match, self.id)
        os.makedirs(game_dir, exist_ok=True)
        path = os.path.join(game_dir, '{}.data'.format(self.id))
        with open(path, 'w', encoding='utf8') as f:
            json.dump(self.to_json(), f, indent=4)
        for r in self.rounds:
            r.save()

    @property
    def is_koth(self):
        return self.map in ['ilios', 'lijiang tower', 'nepal', 'oasis']

    def export(self, path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with open(path, 'w', encoding='utf8') as f:
            json.dump({'map': self.map}, f)


class BaseRound(object):
    def __init__(self, game, id, begin, end):
        self.game = game
        self.begin = begin
        self.end = end
        self.id = id
        self.pause_events = []
        self.left_performances = [PlayerPerformance(x) for x in self.game.left_team.players]
        self.left_npc_performances = {x: NPC(x) for x in npcs}
        self.right_performances = [PlayerPerformance(x) for x in self.game.right_team.players]
        self.right_npc_performances = {x: NPC(x) for x in npcs}

    def to_json(self):
        return {'id': self.id,
                'begin': self.begin,
                'end': self.end,
                'pause_events': self.pause_events,
                'left_performances': [x.to_json() for x in self.left_performances],
                'right_performances': [x.to_json() for x in self.right_performances],
                'left_npc_performances': {k: v.to_json() for k, v in self.left_npc_performances.items()},
                'right_npc_performances': {k: v.to_json() for k, v in self.right_npc_performances.items()}}

    def save(self):
        round_dir = os.path.join(annotations_dir, self.game.match, str(self.game.id), str(self.id))
        os.makedirs(round_dir, exist_ok=True)
        path = os.path.join(round_dir, '{}.data'.format(self.id))
        with open(path, 'w', encoding='utf8') as f:
            json.dump(self.to_json(), f, indent=4)

    def construct_timeline(self, absolute_time=True):
        timeline = []
        if absolute_time:
            offset = self.begin
        else:
            offset = 0
        if absolute_time:
            timeline.append([self.begin, 'MATCH'])
            timeline.append([self.end, 'END'])
        else:
            timeline.append([0, 'MATCH'])
            timeline.append([self.end - self.begin, 'END'])
        left_color = self.game.left_team.color.upper()
        right_color = self.game.right_team.color.upper()

        for k, n in self.left_npc_performances.items():
            for e in n.deaths:
                timeline.append([e, 'DEATH', left_color, '0', k])
        for k, n in self.right_npc_performances.items():
            for e in n.deaths:
                timeline.append([e, 'DEATH', right_color, '0', k])

        for i, p in enumerate(self.left_performances):
            prev = ''
            for s in p.switches:
                timeline.append([s[0] + offset, 'SWITCH', left_color, i + 1, prev, s[1]])
                prev = s[1]
            for d in p.deaths:
                timeline.append([d + offset, 'DEATH', left_color, i + 1, p.hero_at_time(d)])
            for k in p.kills:
                if isinstance(k[1], int):
                    timeline.append([k[0] + offset, 'KILL', left_color, i + 1, p.hero_at_time(k[0]), k[1] + 1,
                                     self.right_performances[k[1]].hero_at_time(k[0]), k[2], ""])
                else:
                    timeline.append([k[0] + offset, 'KILL', left_color, i + 1, p.hero_at_time(k[0]), 0,
                                     k[1], k[2], ""])
            for r in p.revives:
                timeline.append([r[0] + offset, 'REVIVE', left_color, i + 1, p.hero_at_time(r[0]), r[1] + 1,
                                 self.left_performances[r[1]].hero_at_time(r[0]), r[2], ""])
            for ug in p.ult_gains:
                timeline.append([ug + offset, 'ULT_GAIN', left_color, i + 1, p.hero_at_time(ug)])
            for uu in p.ult_uses:
                timeline.append([uu + offset, 'ULT_USE', left_color, i + 1, p.hero_at_time(uu)])

        for i, p in enumerate(self.right_performances):
            prev = ''
            for s in p.switches:
                timeline.append([s[0] + offset, 'SWITCH', right_color, i + 1, prev, s[1]])
                prev = s[1]
            for d in p.deaths:
                timeline.append([d + offset, 'DEATH', right_color, i + 1, p.hero_at_time(d)])
            for k in p.kills:
                if isinstance(k[1], int):
                    timeline.append([k[0] + offset, 'KILL', right_color, i + 1, p.hero_at_time(k[0]), k[1] + 1,
                                     self.left_performances[k[1]].hero_at_time(k[0]), k[2], ""])
                else:
                    timeline.append([k[0] + offset, 'KILL', right_color, i + 1, p.hero_at_time(k[0]), 0,
                                     k[1], k[2], ""])
            for r in p.revives:
                timeline.append([r[0] + offset, 'REVIVE', right_color, i + 1, p.hero_at_time(r[0]), r[1] + 1,
                                 self.right_performances[r[1]].hero_at_time(r[0]), r[2], ""])
            for ug in p.ult_gains:
                timeline.append([ug + offset, 'ULT_GAIN', right_color, i + 1, p.hero_at_time(ug)])
            for uu in p.ult_uses:
                timeline.append([uu + offset, 'ULT_USE', right_color, i + 1, p.hero_at_time(uu)])
        return timeline


class KothRound(BaseRound):
    def __init__(self, game, id, begin, end):
        super(KothRound, self).__init__(game, id, begin, end)
        self.left_point_flips = []
        self.right_point_flips = []

    def to_json(self):
        base = super(KothRound, self).to_json()
        base['left_point_flips'] = self.left_point_flips
        base['right_point_flips'] = self.right_point_flips
        return base

    @staticmethod
    def from_json(data, game):
        r = KothRound(game, data['id'], data['begin'], data['end'])
        r.left_performances = [PlayerPerformance.from_json(x) for x in data['left_performances']]
        r.right_performances = [PlayerPerformance.from_json(x) for x in data['right_performances']]
        return r

    def construct_timeline(self, absolute_time=True):
        timeline = super(KothRound, self).construct_timeline(absolute_time)
        if absolute_time:
            offset = self.begin
        else:
            offset = 0
        left_color = self.game.left_team.color.upper()
        right_color = self.game.right_team.color.upper()

        for p in self.left_point_flips:
            timeline.append([p+offset, 'ATTACK', left_color])

        for p in self.right_point_flips:
            timeline.append([p+offset, 'ATTACK', right_color])
        return sorted(timeline)


class Round(BaseRound):
    def __init__(self, game, id, begin, end, attacking_side):
        super(Round, self).__init__(game, id, begin, end)
        self.attacking_side = attacking_side
        self.point_events = []

    def to_json(self):
        base = super(Round, self).to_json()
        base['attacking_side'] = self.attacking_side
        base['point_events'] = self.point_events
        return base

    @staticmethod
    def from_json(data, game):
        r = Round(game, data['id'], data['begin'], data['end'], data['attacking_side'])
        r.left_performances = [PlayerPerformance.from_json(x) for x in data['left_performances']]
        r.right_performances = [PlayerPerformance.from_json(x) for x in data['right_performances']]
        return r

    def remove_annotations(self, time_point):
        print(repr(time_point), self.round_events)
        self.round_events = [x for x in self.round_events if x[0] != time_point]
        print(repr(time_point), self.round_events)
        self.npc_events = [x for x in self.npc_events if x[0] != time_point]
        for i, p in enumerate(self.left_team.players + self.right_team.players):
            p.switches = [x for x in p.switches if x[0] != time_point]
            p.deaths = [x for x in p.deaths if x != time_point]
            print(p.kills)
            p.kills = [x for x in p.kills if x[0] != time_point]
            print(p.kills)
            p.revives = [x for x in p.revives if x[0] != time_point]
            p.ult_gains = [x for x in p.ult_gains if x != time_point]
            p.ult_uses = [x for x in p.ult_uses if x != time_point]

    def export(self, path):
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        with open(path, 'w', encoding='utf8') as f:
            data = {'left': self.game.left_team.name,
                    'leftTeamID': self.game.left_team.id,
                    'leftTeamColor': self.game.left_team.color,
                    'leftnames': [x.name for x in self.game.left_team.players],
                    'leftIDs': [x.id for x in self.game.left_team.players],
                    'right': self.game.right_team.name,
                    'rightTeamID': self.game.right_team.id,
                    'rightTeamColor': self.game.right_team.color,
                    'rightnames': [x.name for x in self.game.right_team.players],
                    'rightIDs': [x.id for x in self.game.right_team.players],
                    'events': self.construct_timeline(absolute_time=True)}
            json.dump(data, f)

    def construct_timeline(self, absolute_time=True):
        timeline = super(Round, self).construct_timeline(absolute_time)
        if absolute_time:
            offset = self.begin
        else:
            offset = 0
        left_color = self.game.left_team.color.upper()
        right_color = self.game.right_team.color.upper()
        if self.attacking_side == 'left':
            attack_color = left_color
        elif self.attacking_side == 'left':
            attack_color = right_color
        else:
            attack_color = ''
        timeline.append([self.begin, 'ATTACK', attack_color])
        for p in self.point_events:
            timeline.append([p[0] + offset, attack_color, p[1]])
        return sorted(timeline)


def find_offset(m):
    actual_start = actual_starts[m]
    path = os.path.join(annotations_dir, m, '1', '1_1_data.txt')
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    wl_start = int(data['events'][0][0])
    return wl_start - actual_start


def load_teams(m):
    match_dir = os.path.join(annotations_dir, m)
    meta_path = os.path.join(match_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf8') as f:
            meta = json.load(f)
        team1 = Team(meta['team1'], meta['team1_id'], None)
        team1.save()
        for p in meta['team1_players']:
            team1.players.append(Player(p, None))
        team2 = Team(meta['team2'], meta['team2_id'], None)
        team2.save()
        for p in meta['team2_players']:
            team2.players.append(Player(p, None))
    else:
        team1 = Team('', '', '')
        team2 = Team('', '', '')
    return team1, team2


def load_player(id):
    if not id:
        return Player('', '')
    player_dir = os.path.join(annotations_dir, 'players')
    path = os.path.join(player_dir, '{}.data'.format(id))
    with open(path, 'r', encoding='utf8') as f:
        player = Player.from_json(json.load(f))
    return player


def load_team(id):
    if not id:
        return Team('', '', '')
    team_dir = os.path.join(annotations_dir, 'teams')
    path = os.path.join(team_dir, '{}.data'.format(id))
    with open(path, 'r', encoding='utf8') as f:
        team = Team.from_json(json.load(f))
    return team


def load_game(match, id):
    game_dir = os.path.join(annotations_dir, match, id)
    path = os.path.join(game_dir, '{}.data'.format(id))
    if os.path.exists(path):
        with open(path, 'r', encoding='utf8') as f:
            game = Game.from_json(json.load(f))
    else:
        with open(os.path.join(game_dir, 'meta.json'), 'r', encoding='utf8') as df:
            data = json.load(df)
        if 'blue' in data:
            left_team = Team(data['blue'], data['blueTeamID'], 'blue')
            left_team.save()
            left_team.players = [Player(data['bluenames'][i], data['blueIDs'][i]) for i in range(6)]
            for p in left_team.players:
                p.save()

            right_team = Team(data['red'], data['redTeamID'], 'red')
            right_team.save()
            right_team.players = [Player(data['rednames'][i], data['redIDs'][i]) for i in range(6)]
            for p in right_team.players:
                p.save()
        elif 'left' in data:
            left_team = Team(data['left'], data['leftTeamID'], data['leftTeamColor'])
            left_team.save()
            left_team.players = [Player(data['leftnames'][i], data['leftIDs'][i]) for i in range(6)]
            for p in left_team.players:
                p.save()

            right_team = Team(data['right'], data['rightTeamID'], data['rightTeamColor'])
            right_team.save()
            right_team.players = [Player(data['rightnames'][i], data['rightIDs'][i]) for i in range(6)]
            for p in right_team.players:
                p.save()
        else:
            left_team = Team('', '', '')
            left_team.players = [Player('', '') for i in range(6)]
            right_team = Team('', '', '')
            right_team.players = [Player('', '') for i in range(6)]

        game = Game(match, id, data['map'], left_team, right_team)
        game.save()
    return game


def get_round_count(game_dir):
    max_dir = None
    max_file = None
    for d in sorted(os.listdir(game_dir)):
        path = os.path.join(game_dir, d)
        if os.path.isdir(path):
            max_dir = d
        else:
            if not d.endswith('data.txt'):
                continue
            _, r, _ = d.split('_')
            max_file = r
    if max_dir is None and max_file is None:
        return None
    elif max_dir is None:
        return int(max_file)
    return int(max_dir)


def load_round(match, game, round_id):
    round_dir = os.path.join(annotations_dir, match, game.id, str(round_id))
    if os.path.exists(round_dir):
        path = os.path.join(round_dir, '{}.data'.format(round_id))
        with open(path, 'r', encoding='utf8') as f:
            if game.is_koth:
                r = KothRound.from_json(json.load(f), game)
                r.game = game
            else:
                r = Round.from_json(json.load(f), game)
                r.game = game
    else:
        offset = 0
        if match in actual_starts:
            offset = find_offset(match)
        round_path = os.path.join(annotations_dir, match, game.id, '{}_{}_data.txt'.format(game.id, round_id))
        with open(round_path, 'r', encoding='utf8') as df:
            data = json.load(df)
        events = data['events']
        left_team_color = game.left_team.color
        right_team_color = game.right_team.color
        begin, end = events[0][0], events[-1][0]
        if game.is_koth:
            r = KothRound(game, round_id, begin, end)
        else:
            attacking_side = None
            for e in events:
                if e[1] == 'ATTACK':
                    color = e[2].lower()
                    if color == left_team_color:
                        attacking_side = 'left'
                    else:
                        attacking_side = 'right'
                    break
            r = Round(game, round_id, begin, end, attacking_side)
        for e in events:
            timestamp = int(e[0]) - offset
            event_type = e[1]
            if event_type == 'MATCH':
                pass
            elif event_type == 'END':
                pass
            elif event_type == 'ATTACK':
                timestamp -= r.begin
                if game.is_koth:
                    if e[2].lower() == left_team_color:
                        r.left_point_flips.append(timestamp)
                    else:
                        r.right_point_flips.append(timestamp)
            elif event_type == 'POINTS':
                timestamp -= r.begin
                if not game.is_koth:
                    r.point_events.append((timestamp, e[2]))
            elif event_type in ['PAUSE', 'UNPAUSE']:
                timestamp -= r.begin
                r.pause_events.append((timestamp, e[1]))
            elif event_type in ['SWITCH', 'DEATH', 'ULT_GAIN', 'ULT_USE', 'KILL', 'REVIVE']:
                timestamp -= r.begin
                team = e[2].lower()
                player_id = int(e[3]) - 1
                if event_type == 'SWITCH':
                    new_hero = e[-1]
                    if team == left_team_color:

                        r.left_performances[player_id].add_switch(timestamp, new_hero)
                    else:
                        r.right_performances[player_id].add_switch(timestamp, new_hero)
                elif event_type == 'ULT_GAIN':
                    if team == left_team_color:
                        r.left_performances[player_id].add_ult_gain(timestamp)
                    else:
                        r.right_performances[player_id].add_ult_gain(timestamp)
                elif event_type == 'ULT_USE':
                    if team == left_team_color:
                        r.left_performances[player_id].add_ult_use(timestamp)
                    else:
                        r.right_performances[player_id].add_ult_use(timestamp)
                elif event_type == 'DEATH':
                    if player_id >= 0:
                        if team == left_team_color:
                            r.left_performances[player_id].add_death(timestamp)
                        else:
                            r.right_performances[player_id].add_death(timestamp)
                    else:
                        character = e[4]
                        if team == left_team_color:
                            r.left_npc_performances[character].add_death(timestamp)
                        else:
                            r.right_npc_performances[character].add_death(timestamp)
                elif event_type == 'KILL':
                    killed_player_id = int(e[5]) - 1
                    if killed_player_id < 0:
                        killed_player_id = e[6]
                    method = e[7] if e[7] else 'regular'
                    if team == left_team_color:
                        r.left_performances[player_id].add_kill(timestamp, killed_player_id, method)
                    else:
                        r.right_performances[player_id].add_kill(timestamp, killed_player_id, method)
                    if team == left_team_color:
                        team = right_team_color
                    else:
                        team = left_team_color
                    player_id = int(e[5]) - 1
                    if player_id >= 0:
                        if team == left_team_color:
                            r.left_performances[player_id].add_death(timestamp)
                        else:
                            r.right_performances[player_id].add_death(timestamp)
                    else:
                        npc = e[6]
                        if team == left_team_color:
                            r.left_npc_performances[npc].add_death(timestamp)
                        else:
                            r.right_npc_performances[npc].add_death(timestamp)
                elif event_type == 'REVIVE':
                    killed_player_id = int(e[5]) - 1
                    method = e[7] if e[7] else 'regular'
                    if team == left_team_color:
                        r.left_performances[player_id].revives.append((timestamp, killed_player_id, method))
                    else:
                        r.right_performances[player_id].revives.append((timestamp, killed_player_id, method))
        r.save()
    return r

def parse_match(m):
    match_dir = os.path.join(annotations_dir, m)
    games = []
    for d in os.listdir(match_dir):
        game_dir = os.path.join(match_dir, d)
        if not os.path.isdir(game_dir):
            continue
        game = load_game(m, d)
        num_rounds = get_round_count(game_dir)

        for i in range(1,num_rounds+1):
            r = load_round(m, game, i)

            game.rounds.append(r)
        games.append(game)
    return games


if __name__ == '__main__':
    for m in test_files:
        games = parse_match(m)
        for i, g in enumerate(games):
            i += 1
            for j, r in enumerate(g.rounds):
                j += 1
                data_path = os.path.join(annotations_dir, m, str(i), '{}_{}_data.txt'.format(i, j))
                with open(data_path, 'r', encoding='utf8') as df:
                    data = json.load(df)
                events = data['events']
                assert events == r.construct_timeline()
