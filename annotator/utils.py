import os
import subprocess
import requests
import numpy as np

from threading import Thread
from queue import Queue, Empty
import cv2

import annotator.config as config
from annotator.game_values import STATUS_SET

def get_local_file(r):
    directory = config.vod_directory
    vod_link = r['stream_vod']['vod_link']
    out_template = '{}.%(ext)s'.format(r['stream_vod']['id'])
    print(vod_link)
    if vod_link[0] == 'twitch':
        template = 'https://www.twitch.tv/videos/{}'
    subprocess.call(['youtube-dl', '-F', template.format(vod_link[1]), ], cwd=directory)
    for f in ['720p', '720p30']:
        subprocess.call(['youtube-dl', template.format(vod_link[1]), '-o', out_template, '-f', f], cwd=directory)


def get_local_vod(v):
    directory = config.vod_directory
    out_template = '{}.%(ext)s'.format(v['id'])
    print('DOWNLOADING', v)
    subprocess.call(['youtube-dl', '-F', v['url'], ], cwd=directory)
    for f in ['720p', '720p60' '720p30', '22', 'best']:
        subprocess.call(['youtube-dl', v['url'], '-o', out_template, '-f', f], cwd=directory)


def calculate_hero_boundaries(player_name):
    value = 0
    for c in player_name:
        if c in config.CHAR_VALUES:
            value += config.CHAR_VALUES[c]
        else:
            value += config.CHAR_VALUES['default']
    value += len(player_name) - 1
    right_boundary = value + config.PLATE_NAME_LEFT_MARGIN + config.PLATE_RIGHT_MARGIN
    left_boundary = right_boundary + config.PLATE_PORTRAIT_WIDTH
    return left_boundary, right_boundary


def calculate_ability_boundaries(left_hero_boundary, ability):
    if ability in ['primary', 'n/a']:
        width = config.PRIMARY_ABILITY_WIDTH
    else:
        width = config.SPECIAL_ABILITY_WIDTH
    right_boundary = left_hero_boundary + config.PLATE_LEFT_MARGIN
    left_boundary = right_boundary + width
    return left_boundary, right_boundary


def calculate_first_hero_boundaries(left_ability_boundary, num_assists):
    assist_width = config.PLATE_ASSIST_WIDTH * num_assists
    assist_width += config.PLATE_LEFT_MARGIN + config.PLATE_ASSIST_MARGIN * num_assists
    right_boundary = left_ability_boundary + assist_width
    left_boundary = right_boundary + config.PLATE_PORTRAIT_WIDTH
    return left_boundary, right_boundary


def calculate_assist_boundaries(left_ability_boundary, num_assists):
    assist_width = config.PLATE_ASSIST_WIDTH * num_assists + config.PLATE_ASSIST_MARGIN * num_assists
    assist_right = left_ability_boundary + config.PLATE_LEFT_MARGIN
    assist_left = assist_right + assist_width
    return assist_left, assist_right


def calculate_first_player_boundaries(left_hero_boundary, player_name):
    value = 0
    for c in player_name:
        if c in config.CHAR_VALUES:
            value += config.CHAR_VALUES[c]
        else:
            value += config.CHAR_VALUES['default']
    value += len(player_name) - 1
    right_boundary = left_hero_boundary + config.PLATE_NAME_LEFT_MARGIN
    left_boundary = right_boundary + value + config.PLATE_RIGHT_MARGIN
    return left_boundary, right_boundary




def get_vod_path(v):
    vod_path = os.path.join(config.vod_directory, '{}.mp4'.format(v['id']))
    return vod_path


def get_local_path(r):
    match_directory = os.path.join(config.local_directory, str(r['game']['match']['wl_id']))
    game_directory = os.path.join(match_directory, str(r['game']['game_number']))
    game_path = os.path.join(game_directory, '{}.mp4'.format(r['game']['game_number']))
    if os.path.exists(game_path):
        return game_path
    match_path = os.path.join(match_directory, '{}.mp4'.format(r['game']['match']['wl_id']))
    if os.path.exists(match_path):
        return match_path


def look_up_player_state(side, index, time, states, has_status=True):
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
    data['status'] = 'normal'
    if has_status:
        for s in STATUS_SET:
            if not s:
                continue
            ind = np.searchsorted(states[s+'_array'], time, side="right")
            if ind == len(states[s]):
                ind -= 1
            if s in ['asleep', 'frozen', 'hacked', 'stunned']:
                if not states[s][ind]['status'].startswith('not_'):
                    data['status'] = s
            else:
                data[s] = states[s][ind]['status']
    else:
        for x in ['antiheal', 'immortal']:
            data[x] = 'not_' + x

    ind = np.searchsorted(states['hero_array'], time, side="right")
    if ind == len(states['hero']):
        ind -= 1
    data['hero'] = states['hero'][ind]['hero']['name'].lower()
    data['switch'] = 'not_switch'
    if states['hero'][ind]['begin']!= 0 and time - states['hero'][ind]['begin'] < 1.2:
        data['switch'] = 'switch'
    data['player'] = states['player'].lower()
    return data


def look_up_single_round_state(time, states, identifier):
    data = {}
    for t in states[identifier]:
        if t['begin'] <= time < t['end']:
            data[identifier] = t['status']
            break
    else:
        data[identifier] = states[identifier][-1]['status']
    return data


def look_up_game_state(time, states):
    data = {}
    for k in ['game', 'spectator_mode', 'left', 'right']:
        ind = np.searchsorted(states['{}_array'.format(k)], time, side="right")
        if ind == len(states[k]):
            ind -= 1
        data[k] = states[k][ind]['status']
    return data


def look_up_round_state(time, states):
    data = {}
    for k in ['overtime', 'pause', 'replay', 'smaller_window']:
        for t in states[k]:
            if t['begin'] <= time < t['end']:
                data[k] = t['status']
                break
        else:
            data[k] = states[k][-1]['status']
    data['zoomed_bars'] = {}
    for s in ['left', 'right']:
        for t in states['zoomed_bars'][s]:
            if t['begin'] <= time < t['end']:
                data['zoomed_bars'][s] = t['status']
                break
        else:
            data['zoomed_bars'][s] = states['zoomed_bars'][s][-1]['status']

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


class FileVideoStream(object):
    def __init__(self, path, begin, end, time_step, queueSize=100, short_time_steps=None, real_begin=None):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        print(path)
        self.stream = cv2.VideoCapture(path)
        self.short_time_steps = short_time_steps
        if self.short_time_steps is None:
            self.short_time_steps = []
        self.fps = round(self.stream.get(cv2.CAP_PROP_FPS), 2)
        self.stopped = False
        self.begin = round(begin, 1)
        self.end = end
        self.frame_shift = 0
        self.ms_shift = 53
        for p in ['2288', '2287']:
            if p in path:
                self.frame_shift = 10
                self.ms_shift = 400
                break
        if self.end == 0:
            self.end = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps) - 5
        self.stream.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        print('DURATION', self.stream.get(cv2.CAP_PROP_POS_MSEC))
        self.time_step = time_step
        self.real_begin = real_begin
        if self.real_begin is not None:
            self.real_begin = round(real_begin, 1)


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
                time_point = round(time_point, 1)
                #print(self.begin, time_point)
                frame_number = int(time_point * self.fps)
                frame_number -= self.frame_shift
                if frame_number >= self.stream.get(cv2.CAP_PROP_FRAME_COUNT):
                    self.stop()
                    return
                #self.stream.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                self.stream.set(cv2.CAP_PROP_POS_MSEC, int(time_point*1000)-self.ms_shift)
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
                ts = self.time_step
                if self.short_time_steps:
                    for interval in self.short_time_steps:
                        if interval['begin'] <= time_point <= interval['end']:
                            ts = 0.1
                            break

                time_point += ts
                frame_ind += 1
                if time_point >= self.end:
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
