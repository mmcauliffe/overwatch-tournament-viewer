import os
import subprocess
import requests
import numpy as np

from threading import Thread
from queue import Queue, Empty
import cv2

local_directory = r'E:\Data\Overwatch\raw_data\annotations\matches'
vod_directory = r'E:\Data\Overwatch\raw_data\vods'

site_url = 'http://localhost:8000/'

api_url = site_url + 'annotator/api/'


def get_local_file(r):
    directory = vod_directory
    vod_link = r['stream_vod']['vod_link']
    out_template = '{}.%(ext)s'.format(r['stream_vod']['id'])
    print(vod_link)
    if vod_link[0] == 'twitch':
        template = 'https://www.twitch.tv/videos/{}'
    subprocess.call(['youtube-dl', '-F', template.format(vod_link[1]), ], cwd=directory)
    for f in ['720p', '720p30']:
        subprocess.call(['youtube-dl', template.format(vod_link[1]), '-o', out_template, '-f', f], cwd=directory)


def get_local_vod(v):
    directory = vod_directory
    out_template = '{}.%(ext)s'.format(v['id'])
    subprocess.call(['youtube-dl', '-F', v['url'], ], cwd=directory)
    for f in ['720p', '720p30']:
        subprocess.call(['youtube-dl', v['url'], '-o', out_template, '-f', f], cwd=directory)


def calculate_hero_boundaries(player_name):
    value = 0
    for c in player_name:
        if c in CHAR_VALUES:
            value += CHAR_VALUES[c]
        else:
            value += CHAR_VALUES['default']
    value += len(player_name) - 1
    right_boundary = value + PLATE_NAME_LEFT_MARGIN + PLATE_RIGHT_MARGIN
    left_boundary = right_boundary + PLATE_PORTRAIT_WIDTH
    return left_boundary, right_boundary


def calculate_ability_boundaries(left_hero_boundary, ability):
    if ability in ['primary', 'n/a']:
        width = PRIMARY_ABILITY_WIDTH
    else:
        width = SPECIAL_ABILITY_WIDTH
    right_boundary = left_hero_boundary + PLATE_LEFT_MARGIN
    left_boundary = right_boundary + width
    return left_boundary, right_boundary


def calculate_first_hero_boundaries(left_ability_boundary, num_assists):
    assist_width = PLATE_ASSIST_WIDTH * num_assists
    assist_width += PLATE_LEFT_MARGIN + PLATE_ASSIST_MARGIN * num_assists
    right_boundary = left_ability_boundary + assist_width
    left_boundary = right_boundary + PLATE_PORTRAIT_WIDTH
    return left_boundary, right_boundary


def calculate_assist_boundaries(left_ability_boundary, num_assists):
    assist_width = PLATE_ASSIST_WIDTH * num_assists + PLATE_ASSIST_MARGIN * num_assists
    assist_right = left_ability_boundary + PLATE_LEFT_MARGIN
    assist_left = assist_right + assist_width
    return assist_left, assist_right


def calculate_first_player_boundaries(left_hero_boundary, player_name):
    value = 0
    for c in player_name:
        if c in CHAR_VALUES:
            value += CHAR_VALUES[c]
        else:
            value += CHAR_VALUES['default']
    value += len(player_name) - 1
    right_boundary = left_hero_boundary + PLATE_NAME_LEFT_MARGIN
    left_boundary = right_boundary + value + PLATE_RIGHT_MARGIN
    return left_boundary, right_boundary


PLATE_ASSIST_WIDTH = 16
PLATE_ASSIST_MARGIN = 3
PLATE_LEFT_MARGIN = 3
PLATE_NAME_LEFT_MARGIN = 6
PLATE_RIGHT_MARGIN = 9
PLATE_PORTRAIT_WIDTH = 36
PRIMARY_ABILITY_WIDTH = 25
SPECIAL_ABILITY_WIDTH = 50

CHAR_VALUES = {'a': 6,
               'b': 5,
               'c': 6,
               'd': 7,
               'e': 4,
               'f': 4,
               'g': 8,
               'h': 7,
               'i': 2,
               'j': 4,
               'k': 6,
               'l': 4,
               'm': 8,
               'n': 7,
               'o': 8,
               'p': 5,
               'q': 8,
               'r': 5,
               's': 5,
               't': 4,
               'u': 7,
               'v': 6,
               'w': 9,
               'x': 7,
               'y': 6,
               'z': 6,
               'default': 6
               }

stage_2_shift = 25

BOX_PARAMETERS = {
    'O': {
        'MID': {
            'X': 520,
            'Y': 35,
            'HEIGHT': 84,
            'WIDTH': 240,
        },
        'KILL_FEED': {
            'X': 1280 - 20 - 250,
            'Y': 112,
            'WIDTH': 256,
            'HEIGHT': 256
        },
        'KILL_FEED_SLOT': {
            'X': 1280 - 20 - 248,
            'Y': 112,
            'WIDTH': 248,
            'HEIGHT': 32,
            'MARGIN': 3
        },
        'LEFT': {
            'X': 34,
            'Y': 42,
            'WIDTH': 64,
            'HEIGHT': 64,
            'MARGIN': 6,
        },
        'RIGHT': {
            'X': 830,
            'Y': 42,
            'WIDTH': 64,
            'HEIGHT': 64,
            'MARGIN': 6,
        },
        'REPLAY': {
            'X': 105,
            'Y': 110,
            'WIDTH': 210,
            'HEIGHT': 60,
        },
        'PAUSE': {
            'X': 550,
            'Y': 310,
            'WIDTH': 150,
            'HEIGHT': 40,
        }
    }}

BOX_PARAMETERS['W'] = {
    'MID': {
        'X': 520,
        'Y': 34,
        'HEIGHT': 84,
        'WIDTH': 240,
    },
    'KILL_FEED': {
        'X': 1280 - 20 - 248,
        'Y': 112,
        'WIDTH': 256,
        'HEIGHT': 256
    },
    'KILL_FEED_SLOT': {
        'X': 1280 - 20 - 248,
        'Y': 112,
        'WIDTH': 248,
        'HEIGHT': 32,
        'MARGIN': 2
    },
    'LEFT': {
        'X': 34,
        'Y': 42,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 6,
    },
    'RIGHT': {
        'X': 830,
        'Y': 42,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 6,
    },
    'REPLAY': {
        'X': 125,
        'Y': 165,
        'WIDTH': 210,
        'HEIGHT': 60,
    },
    'PAUSE': {
        'X': 550,
        'Y': 300,
        'WIDTH': 190,
        'HEIGHT': 70,
    }
}

BOX_PARAMETERS['2'] = {
    'MID': {
        'X': 520,
        'Y': 60,
        'HEIGHT': 84,
        'WIDTH': 240,
    },
    'KILL_FEED': {
        'X': 1280 - 20 - 248,
        'Y': 136,
        'WIDTH': 248,
        'HEIGHT': 256
    },
    'KILL_FEED_SLOT': {
        'X': 1280 - 20 - 248,
        'Y': 136,
        'WIDTH': 248,
        'HEIGHT': 32,
        'MARGIN': 2
    },
    'LEFT': {
        'X': 36,
        'Y': 67,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 7,
    },
    'RIGHT': {
        'X': 827,
        'Y': 67,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 7,
    },
    'REPLAY': {
        'X': 145,
        'Y': 150,
        'WIDTH': 210,
        'HEIGHT': 60,
    },
    'PAUSE': {
        'X': 530,
        'Y': 300,
        'WIDTH': 210,
        'HEIGHT': 80,
    }
}

BOX_PARAMETERS['A'] = {  # Black borders around video feed
    'MID': {
        'X': 490,
        'Y': 45,
        'HEIGHT': 84,
        'WIDTH': 230,
    },
    'KILL_FEED': {
        'X': 950,
        'Y': 115,
        'WIDTH': 270,
        'HEIGHT': 205
    },
    'LEFT': {
        'X': 51,
        'Y': 45,
        'WIDTH': 67,
        'HEIGHT': 55,
        'MARGIN': 1,
    },
    'RIGHT': {
        'X': 825,
        'Y': 45,
        'WIDTH': 67,
        'HEIGHT': 55,
        'MARGIN': 1,
    },
    'REPLAY': {
        'X': 115,
        'Y': 120,
        'WIDTH': 150,
        'HEIGHT': 40,
    },
    'PAUSE': {
        'X': 550,
        'Y': 300,
        'WIDTH': 190,
        'HEIGHT': 70,
    }
}


def get_train_rounds():
    url = api_url + 'train_rounds/'
    r = requests.get(url)
    return r.json()


def get_train_rounds_plus():
    url = api_url + 'train_rounds_plus/'
    r = requests.get(url)
    return r.json()


def get_train_vods():
    url = api_url + 'train_vods/'
    r = requests.get(url)
    return r.json()


def get_annotate_rounds():
    url = api_url + 'annotate_rounds/'
    r = requests.get(url)
    return r.json()


def get_annotate_vods():
    url = api_url + 'annotate_vods/'
    r = requests.get(url)
    return r.json()


def get_player_states(round_id):
    url = api_url + 'rounds/{}/player_states/'.format(round_id)
    r = requests.get(url)
    data = r.json()
    for side, d in data.items():
        for ind, v in d.items():
            for k in ['ult', 'alive', 'hero']:
                data[side][ind]['{}_array'.format(k)] = np.array([x['end'] for x in v[k]])
    return data


def get_round_states(round_id):
    url = api_url + 'rounds/{}/round_states/'.format(round_id)
    r = requests.get(url)
    return r.json()


def get_kf_events(round_id):
    url = api_url + 'rounds/{}/kill_feed_events/'.format(round_id)
    r = requests.get(url)
    return r.json()


def get_hero_list():
    url = api_url + 'heroes/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_player_list():
    url = api_url + 'train_players/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_color_list():
    url = api_url + 'team_colors/'
    r = requests.get(url)
    return sorted(set(x.lower() for x in r.json()))


def get_spectator_modes():
    url = api_url + 'spectator_modes/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_maps():
    url = api_url + 'maps/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_npc_list():
    url = api_url + 'npcs/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_map_modes():
    url = api_url + 'map_modes/'
    r = requests.get(url)
    return sorted(set(x.lower() for x in r.json()))


def get_ability_list():
    ability_set = set()
    url = api_url + 'abilities/damaging_abilities/'
    r = requests.get(url)
    resp = r.json()
    for a in resp:
        ability_set.add(a['name'].lower())
        if a['headshot_capable']:
            ability_set.add(a['name'].lower() + ' headshot')
    url = api_url + 'abilities/reviving_abilities/'
    r = requests.get(url)
    resp = r.json()
    for a in resp:
        ability_set.add(a['name'].lower())
    return ability_set

def upload_game(data):
    url = api_url + 'games/'
    print(data)
    resp = requests.post(url, json=data)
    if resp.status_code != 200:
        raise Exception
    print(resp)


def update_annotations(data, round_id):
    to_send = {'player_states': {},
               'kill_feed': data['kill_feed'],
               'replays': data['replays'],
               'pauses': data['pauses'],
               'ignore_switches': data['ignore_switches']
               }
    for k, v in data['player'].items():
        k = '{}_{}'.format(*k)
        to_send['player_states'][k] = {}
        switches = v.generate_switches()
        ug, uu = v.generate_ults()

        to_send['player_states'][k]['switches'] = switches
        to_send['player_states'][k]['ult_gains'] = ug
        to_send['player_states'][k]['ult_uses'] = uu
    url = api_url + 'annotate_rounds/{}/'.format(round_id)
    print(to_send)
    resp = requests.put(url, json=to_send)
    if resp.status_code != 200:
        raise Exception
    print(resp)

def get_character_set(player_set):
    chars = set()
    for p in player_set:
        chars.update(p)
    return sorted(chars)

MAP_SET = get_maps()

HERO_SET = get_hero_list() + get_npc_list()

ABILITY_SET = sorted(get_ability_list())

COLOR_SET = get_color_list()

SPECTATOR_MODES = get_spectator_modes()

PLAYER_SET = get_player_list()

PLAYER_CHARACTER_SET = get_character_set(PLAYER_SET)

MAP_MODE_SET = get_map_modes()


def get_vod_path(v):
    print(v)
    vod_path = os.path.join(vod_directory, '{}.mp4'.format(v['id']))
    return vod_path


def get_local_path(r):
    match_directory = os.path.join(local_directory, str(r['game']['match']['wl_id']))
    game_directory = os.path.join(match_directory, str(r['game']['game_number']))
    game_path = os.path.join(game_directory, '{}.mp4'.format(r['game']['game_number']))
    if os.path.exists(game_path):
        return game_path
    match_path = os.path.join(match_directory, '{}.mp4'.format(r['game']['match']['wl_id']))
    if os.path.exists(match_path):
        return match_path


def look_up_player_state(side, index, time, states):
    states = states[side][str(index)]

    data = {}
    ind = np.searchsorted(states['ult_array'], time, side="right")
    if ind == len(states['ult']):
        ind -= 1
    data['ult'] = states['ult'][ind]['status']

    ind = np.searchsorted(states['alive_array'], time, side="right")
    if ind == len(states['alive']):
        ind -= 1
    data['alive'] = states['alive'][ind]['status']

    ind = np.searchsorted(states['hero_array'], time, side="right")
    if ind == len(states['hero']):
        ind -= 1
    data['hero'] = states['hero'][ind]['hero']['name'].lower()

    data['player'] = states['player'].lower()
    return data


def look_up_round_state(time, states):
    data = {}
    for t in states['overtimes']:
        if t['begin'] <= time < t['end']:
            data['overtime'] = t['status']
            break
    else:
        data['overtime'] = states['overtimes'][-1]['status']
    for t in states['pauses']:
        if t['begin'] <= time < t['end']:
            data['pause'] = t['status']
            break
    else:
        data['pause'] = states['pauses'][-1]['status']
    for t in states['replays']:
        if t['begin'] <= time < t['end']:
            data['replay'] = t['status']
            break
    else:
        data['replay'] = states['replays'][-1]['status']
    for t in states['point_status']:
        if t['begin'] <= time < t['end']:
            data['point_status'] = t['status']
            break
    else:
        data['point_status'] = 'n/a'
    return data


def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


class FileVideoStreamRange:
    def __init__(self, path, begin, ranges, time_step, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.stopped = False
        self.begin = begin
        self.ranges = ranges
        self.time_step = time_step

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        for r in self.ranges:
            time_point = r['begin']
            # keep looping infinitely
            while True:
                # if the thread indicator variable is set, stop the
                # thread
                if self.stopped:
                    return

                # otherwise, ensure the queue has room in it
                if not self.Q.full():
                    # read the next frame from the file
                    frame_number = int(round(round(time_point + self.begin, 1) * self.fps))
                    self.stream.set(1, frame_number)
                    (grabbed, frame) = self.stream.read()
                    # if the `grabbed` boolean is `False`, then we have
                    # reached the end of the video file
                    if not grabbed:
                        self.stop()
                        return

                    # add the frame to the queue
                    self.Q.put((frame, time_point))
                    time_point += self.time_step
                    time_point = round(time_point, 1)
                    if time_point > r['end']:
                        break
        self.stop()

    def read(self):
        # return next frame in the queue
        if self.stopped and self.Q.qsize() == 0:
            raise Empty
        return self.Q.get()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0


class FileVideoStream:
    def __init__(self, path, begin, end, time_step, queueSize=128, real_begin=None):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.stopped = False
        self.begin = begin
        self.end = end
        if self.end == 0:
            self.end = self.stream.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self.time_step = time_step
        self.real_begin = real_begin

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        time_point = self.begin
        frame_ind = 0
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                frame_number = int(round(time_point * self.fps))
                self.stream.set(1, frame_number)
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                # add the frame to the queue
                if self.real_begin is not None:
                    beg = self.real_begin
                else:
                    beg = self.begin
                self.Q.put((frame, round(time_point - beg, 1)))
                time_point += self.time_step
                frame_ind += 1
                if time_point > self.end:
                    self.stop()
                    return

    def read(self):
        # return next frame in the queue
        if self.stopped and self.Q.qsize() == 0:
            raise Empty
        return self.Q.get()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0


def get_event_ranges(events, end):
    window = 8
    ranges = []
    for e in events:
        if not ranges or e['time_point'] > ranges[-1]['end']:
            ranges.append({'begin': e['time_point'], 'end': round(e['time_point'] + window, 1)})
        elif e['time_point'] <= ranges[-1]['end']:
            ranges[-1]['end'] = round(e['time_point'] + window, 1)
    if ranges[-1]['end'] > end:
        ranges[-1]['end'] = end
    print(ranges)
    return ranges


def construct_kf_at_time(events, time):
    window = 7.3
    possible_kf = []
    for e in events:
        if e['time_point'] > time + 0.25:
            break
        elif e['time_point'] > time:
            possible_kf.insert(0, {'time_point': e['time_point'],
                                   'first_hero': 'n/a', 'first_color': 'n/a', 'first_player': 'n/a', 'ability': 'n/a',
                                   'headshot': 'n/a',
                                   'second_hero': 'n/a',
                                   'second_color': 'n/a', 'second_player': 'n/a'})
        if time - window <= e['time_point'] <= time:
            for k, v in e.items():
                if isinstance(v, str):
                    e[k] = v.lower()
                # if 'color' in k:
                #    if e[k] != 'white':
                #        e[k] = 'nonwhite'
            possible_kf.append(e)
    possible_kf = sorted(possible_kf, key=lambda x: -1 * x['time_point'])
    return possible_kf[:6]
