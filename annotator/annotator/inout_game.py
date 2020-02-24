import os
import json
import cv2
import numpy as np
import re
import torch
import itertools
import datetime
import time
from collections import defaultdict, Counter
from annotator.config import BOX_PARAMETERS, offsets
from annotator.annotator.classes import InGameAnnotator, MidAnnotator, PlayerNameAnnotator, PlayerStatusAnnotator, KillFeedAnnotator
from annotator.annotator.classes.base import filter_statuses

from annotator.utils import get_local_vod, \
    get_local_path,  FileVideoStream, Empty, get_vod_path, check_vod_resolution, get_duration
from annotator.api_requests import get_annotate_vods_in_out_game, upload_annotated_in_out_game

frames_per_seq = 100

working_dir = r'N:\Data\Overwatch\models'
player_model_dir = os.path.join(working_dir, 'player_status')
player_ocr_model_dir = os.path.join(working_dir, 'player_ocr_test')
kf_ctc_model_dir = os.path.join(working_dir, 'kill_feed_ctc')
game_model_dir = os.path.join(working_dir, 'game_test')
game_detail_model_dir = os.path.join(working_dir, 'game_detail')
mid_model_dir = os.path.join(working_dir, 'mid')

annotation_dir = r'N:\Data\Overwatch\annotations'
oi_annotation_dir = r'N:\Data\Overwatch\oi_annotations'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_basic(v):
    info = {}
    pattern = r'''(.*[-,:\]] )?(?P<team_one>[-\w '?.]+) [vV][sS][.]? (?P<team_two>[-\w '?.]+)( [-:].*)?'''
    m = re.match(pattern, v['title'])
    if m is not None:
        info['team_one'] = m.group('team_one')
        info['team_two'] = m.group('team_two')
    return info


def extract_info(v):
    owl_mapping = {'DAL': 'Dallas Fuel',
                   'PHI': 'Philadelphia Fusion', 'SEO': 'Seoul Dynasty',
                   'LDN': 'London Spitfire', 'SFS': 'San Francisco Shock', 'HOU': 'Houston Outlaws',
                   'BOS': 'Boston Uprising', 'VAL': 'Los Angeles Valiant', 'GLA': 'Los Angeles Gladiators',
                   'FLA': 'Florida Mayhem', 'SHD': 'Shanghai Dragons', 'NYE': 'New York Excelsior',
                   'PAR': 'Paris Eternal', 'TOR': 'Toronto Defiant', 'WAS': 'Washington Justice',
                   'VAN': 'Vancouver Titans', 'CDH': 'Chengdu Hunters', 'HZS': 'Hangzhou Spark',
                   'ATL': 'Atlanta Reign', 'GZC': 'Guangzhou Charge',
                   'CHE': 'Chengdu Hunters', 'GUA': 'Guangzhou Charge', 'HAN': 'Hangzhou Spark'}
    channel = v['channel']['name']
    info = {}
    v['title'] = v['title'].strip()
    if channel.lower() in ['overwatchcontenders', 'overwatchcontendersbr']:
        pattern = r'''(?P<team_one>[-\w ']+) (vs|V) (?P<team_two>[-\w ']+) \| (?P<desc>[\w ]+) Game (?P<game_num>\d) \| ((?P<sub>[\w :]+) \| )?(?P<main>[\w ]+)'''
        m = re.match(pattern, v['title'])
        if m is not None:
            print(m.groups())
            info['team_one'] = m.group('team_one')
            info['team_two'] = m.group('team_two')
            info['description'] = m.group('desc')
            info['game_number'] = m.group('game_num')
            sub = m.group('sub')
            main = m.group('main')
            info['event'] = main
            if sub is not None:
                info['event'] += ' - ' + sub
    elif channel.lower() == 'overwatch contenders':
        pattern = r'''(?P<team_one>[-\w .']+) (vs|V) (?P<team_two>[-\w .']+) (\(Part)?.*'''
        m = re.match(pattern, v['title'])
        if m is not None:
            info['team_one'] = m.group('team_one')
            info['team_two'] = m.group('team_two')
    elif channel.lower() == 'overwatchleague':
        pattern = r'Game [#]?(\d+) (\w+) @ (\w+) \| ([\w ]+)'
        m = re.match(pattern, v['title'])
        if m is not None:

            game_number, team_one, team_two, desc = m.groups()
            info['team_one'] = owl_mapping[team_one]
            info['team_two'] = owl_mapping[team_two]
            info['game_number'] = game_number
    elif channel.lower() =='owlettournament':
        pattern = r'''.* - (?P<team_one>[-\w ']+) (vs[.]?|V) (?P<team_two>[-\w ']+)'''

        m = re.match(pattern, v['title'])
        if m is not None:
            info['team_one'] = m.group('team_one')
            info['team_two'] = m.group('team_two')
    elif channel.lower() =='owlet tournament':
        pattern = r'''.*: (?P<team_one>[-\w ']+) (vs[.]?|V) (?P<team_two>[-\w ']+)'''

        m = re.match(pattern, v['title'])
        if m is not None:
            info['team_one'] = m.group('team_one')
            info['team_two'] = m.group('team_two')
    elif channel.lower() == 'rivalcade':
        pattern = r'''.*, (?P<team_one>[-\w. '?]+) (vs[.]?|VS) (?P<team_two>[-\w. '?]+)'''
        m = re.match(pattern, v['title'])
        if m is not None:
            info['team_one'] = m.group('team_one')
            info['team_two'] = m.group('team_two')
    if not info:
        info = extract_basic(v)
    if 'team_one' in info:
        info['team_one'] = info['team_one'].strip()
        info['team_two'] = info['team_two'].strip()
    return info


def annotate_game_or_not(v):
    actual_duration, mode = get_duration(v)
    offset = None
    if v['id'] in offsets:
        offset = offsets[v['id']]
    game_annotator = InGameAnnotator(v['film_format']['code'], v['spectator_mode'].lower(),
                                     game_model_dir, device, use_spec_modes=False)

    display_interval = 50
    if offset:
        game_annotator.begin_time += offset
        display_interval += offset

    fvs = FileVideoStream(get_vod_path(v), 0, 0, game_annotator.time_step, real_begin=0).start()
    time.sleep(5)
    print('begin ingame/outofgame processing')
    while True:
        try:
            frame, time_point = fvs.read()
        except Empty:
            break
        if offset:
            time_point += offset
        frame = frame['frame']
        if (offset and (time_point - offset) % 50 == 0) or time_point % 50 == 0:
            print(time_point)
        game_annotator.process_frame(frame, time_point)
    game_annotator.annotate()
    data, rounds = game_annotator.generate_rounds()
    # Fine tune rounds
    old_time_step = game_annotator.time_step
    game_annotator.time_step = 0.1
    for i, r in enumerate(rounds):
        new_start = round(r['begin'] - old_time_step, 1)
        if new_start < 0:
            new_start = 0
        #print('BOUNDS', new_start, r['begin'])
        if new_start != r['begin']:
            fvs = FileVideoStream(get_vod_path(v), new_start,
                                  r['begin'], 0.1, real_begin=0).start()
            game_annotator.reset(begin_time=new_start)
            game_annotator.reset_status()
            while True:
                try:
                    frame, time_point = fvs.read()
                except Empty:
                    break
                if offset:
                    time_point += offset
                frame = frame['frame']
                game_annotator.process_frame(frame, time_point)

            game_annotator.annotate()
            earliest_game = game_annotator.get_earliest('game', 'game')
            if earliest_game is not None:
                rounds[i]['begin'] = earliest_game
        fvs = FileVideoStream(get_vod_path(v), r['end'] - old_time_step,
                              r['end'] + old_time_step, 0.1, real_begin=0).start()
        game_annotator.reset(begin_time=r['end'] - old_time_step)
        print('BOUNDS', r['end'], r['end'] + old_time_step)
        game_annotator.reset_status()
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            if offset:
                time_point += offset
            frame = frame['frame']
            game_annotator.process_frame(frame, time_point)
        game_annotator.annotate()
        latest_game = game_annotator.get_latest('game', 'game')
        if latest_game is not None:
            rounds[i]['end'] = round(latest_game + 0.1, 1)

        for k in ['pauses', 'replays', 'smaller_windows']:
            for e in r[k]:
                if k == 'pauses' and e['type'] == 'Out of game pause':
                    label = 'not_game'
                elif k == 'pauses' and e['type'] == 'Split screen':
                    label = 'pause_split screen'
                else:
                    label = k[:-1]
                new_start = e['begin'] - old_time_step
                if new_start < 0:
                    new_start = 0
                if new_start != e['begin']:
                    fvs = FileVideoStream(get_vod_path(v), r['begin'] + new_start,
                                          r['begin'] + e['begin'], 0.1, real_begin=r['begin']).start()
                    game_annotator.reset(begin_time=e['begin'] - old_time_step)
                    game_annotator.reset_status()
                    while True:
                        try:
                            frame, time_point = fvs.read()
                        except Empty:
                            break
                        if offset:
                            time_point += offset
                        frame = frame['frame']
                        game_annotator.process_frame(frame, time_point)

                        #cv2.imshow('frame_{}'.format(time_point), frame)
                    game_annotator.annotate()
                    earliest = game_annotator.get_earliest('game', label)
                    if earliest is not None:
                        e['begin'] = earliest

                    #cv2.waitKey()
                    fvs = FileVideoStream(get_vod_path(v), r['begin'] + e['end'] - old_time_step,
                                          r['begin'] + e['end'] + old_time_step, 0.1, real_begin=r['begin']).start()
                    game_annotator.reset(begin_time=e['end'] - old_time_step)
                    game_annotator.reset_status()
                    while True:
                        try:
                            frame, time_point = fvs.read()
                        except Empty:
                            break
                        if offset:
                            time_point += offset
                        frame = frame['frame']
                        game_annotator.process_frame(frame, time_point)
                    game_annotator.annotate()
                    latest = game_annotator.get_latest('game', label)
                    if latest is not None:
                        e['end'] = round(latest + 0.1, 1)
        for k in ['left', 'right']:
            if r[k+ '_zooms'] and 'ZOOMED_LEFT' not in BOX_PARAMETERS[v['film_format']['code']]:
                r[k+'_zooms'] = []
            for e in r[k + '_zooms']:
                new_start = e['begin'] - old_time_step
                if new_start < 0:
                    new_start = 0
                if new_start != e['begin']:
                    fvs = FileVideoStream(get_vod_path(v), r['begin'] + new_start,
                                          r['begin'] + e['begin'], 0.1, real_begin=r['begin']).start()
                    game_annotator.reset(begin_time=e['begin'] - old_time_step)
                    game_annotator.reset_status()
                    while True:
                        try:
                            frame, time_point = fvs.read()
                        except Empty:
                            break
                        if offset:
                            time_point += offset
                        frame = frame['frame']
                        game_annotator.process_frame(frame, time_point)
                    game_annotator.annotate()
                    earliest = game_annotator.get_earliest(k, 'zoom')
                    if earliest is not None:
                        e['begin'] = earliest

                fvs = FileVideoStream(get_vod_path(v), r['begin'] + e['end'] - old_time_step,
                                      r['begin'] + e['end'] + old_time_step, 0.1, real_begin=r['begin']).start()
                game_annotator.reset(begin_time=e['end'] - old_time_step)
                game_annotator.reset_status()
                while True:
                    try:
                        frame, time_point = fvs.read()
                    except Empty:
                        break
                    if offset:
                        time_point += offset
                    frame = frame['frame']
                    game_annotator.process_frame(frame, time_point)
                game_annotator.annotate()
                latest = game_annotator.get_latest(k, 'zoom')
                if latest is not None:
                    e['end'] = round(latest + 0.1, 1)
    return data, rounds

def get_name_begin(r):
    begin = 0
    for sw in r['smaller_windows']:
        if sw['begin'] < 5:
            begin = sw['end'] + 1
            break
    else:
        for replay in r['replays']:
            if replay['begin'] < 5:
                begin = replay['end'] + 1
                break
        else:
            for pause in r['pauses']:
                if pause['begin'] < 5:
                    begin = pause['end'] + 1
                    break
    return begin

def extract_properties(v, r, spectator_mode):
    round_level = ['round_number', 'attacking_side']
    game_level = ['map', 'spectator_mode', 'map_mode']
    print('begin', timestamp(r['begin']))
    print('end', timestamp(r['end']))
    film_format = v['film_format']['code']
    #mid_annotator = MidAnnotator(film_format, mid_model_dir, device,
    #                             spectator_mode, r['map'], r['attacking_side'])
    #mid_annotator.time_step = 1
    name_annotator = PlayerNameAnnotator(film_format, player_ocr_model_dir, device, spectator_mode,debug=False)
    begin = get_name_begin(r)
    begin += r['begin']
    fvs = FileVideoStream(get_vod_path(v), begin, begin + 5, name_annotator.time_step, real_begin=r['begin']).start()
    time.sleep(5)
    print('begin name processing')
    while True:
        try:
            frame, time_point = fvs.read()
        except Empty:
            break
        frame = frame['frame']
        name_annotator.process_frame(frame, time_point)
    name_annotator.annotate()
    names = name_annotator.generate_names(v['teams'])
    #r['mid_map'] = statuses['map']
    #r['map_mode'] = statuses['map_mode']
    #r['round_number'] = statuses['round_number']
    #r['attacking_side'] = statuses['attacking_side']
    r['players'] = names
    return r

def annotate_properties(v, rounds, spectator_mode):
    import time
    for i, r in enumerate(rounds):
        rounds[i] = extract_properties(v, r, spectator_mode)
        print(r)
    #data['map'] = max(m, key=lambda x: m[x])
    return rounds

def timestamp(seconds):
    return str(datetime.timedelta(seconds=seconds))

def print_round_data(r):
    print('-------------')
    print('Round number {}'.format(r['round_number']))
    begin_timestamp = timestamp(r['begin'])
    end_timestamp = timestamp(r['end'])
    print('Begin: {}, {}'.format(begin_timestamp, r['begin']))
    print('End: {}, {}'.format(end_timestamp, r['end']))
    print('Submap: {}'.format(r['submap']))
    print('Left team: {}'.format(', '.join(r['players']['left'].values())))
    print('Right team: {}'.format(', '.join(r['players']['right'].values())))
    print('Pauses: {}'. format(', '.join(timestamp(x['begin'])+'-'+timestamp(x['end']) for x in r['pauses'])))
    print('Replays: {}'. format(', '.join(timestamp(x['begin'])+'-'+timestamp(x['end']) for x in r['replays'])))
    print('Smaller windows: {}'. format(', '.join(timestamp(x['begin'])+'-'+timestamp(x['end']) for x in r['smaller_windows'])))

def print_game_data(data):
    print('-------------')
    if 'games' in data:
        print('Multiple games')
        print('Found {} games'.format(len(data['games'])))
        print('-------------')
        for g in data['games']:
            print('Game {}'.format(g['game_number']))
            print('Map: {}'.format(g['map']))
            print('Left color: {}'.format(g['left_color']))
            print('Right color: {}'.format(g['right_color']))
            for r in g['rounds']:
                print_round_data(r)
    else:
        print('Just one game')
        print('Found {} rounds'.format(len(data['rounds'])))
        for r in data['rounds']:
            print_round_data(r)


def analyze_ingames(vods):
    game_dir = os.path.join(oi_annotation_dir, 'to_check')
    os.makedirs(game_dir, exist_ok=True)
    for v in vods:
        print(v)
        #if v['spectator_mode'].lower() != 'overwatch league':
        #    continue
        #if v['id'] in [3120, 3121, 3122]:
        #    continue
        info= extract_info(v)
        print(info)
        if not info:
            continue
        data = {'vod_id': v['id'], 'team_one': info['team_one'].lower(), 'team_two': info['team_two'].lower()}
        if data['team_one'] == 'gen g esports':
            data['team_one'] = 'gen.g esports'
        if v['type'] == 'G' and 'game_number' in info:
            data['game_number'] = int(info['game_number'])
            data['rounds'] = []
        elif v['type'] == 'M':
            data['games'] = []
        print(data)
        g_data, rounds = annotate_game_or_not(v)
        data.update(g_data)

        annotate_properties(v, rounds, v['spectator_mode'])
        for i, r in enumerate(rounds):
            begin_timestamp = timestamp(r['begin'])
            end_timestamp = timestamp(r['end'])
            print('ROUND: {}-{}'.format(begin_timestamp, end_timestamp))
            for k in r.keys():
                if k in ['begin', 'end']:
                    continue
                if isinstance(r[k], list):
                    print(k)
                    for interval in r[k]:
                        begin_timestamp = timestamp(interval['begin'])
                        end_timestamp = timestamp(interval['end'])
                        print('{}-{}'.format(begin_timestamp, end_timestamp))
                else:
                    print(k, r[k])
        if v['type'] == 'G':
            for i, r in enumerate(rounds):
                r['round_number'] = i + 1
            data['rounds'] = rounds
            m = Counter()
            for r in rounds:
                m[r['map']] += 1
            data['map'] = max(m.keys(), key=lambda x: m[x])
        elif v['type'] == 'M':
            data['games'].append({'game_number': 1, 'rounds': [], 'left_color': data['left_color'], 'right_color': data['right_color']})
            for i, r in enumerate(rounds):

                if data['games'][-1]['rounds'] and r['map'] != data['games'][-1]['rounds'][-1]['map']:
                    r['round_number'] = 1
                    data['games'].append({'game_number': data['games'][-1]['game_number'] + 1, 'map': r['map'],
                                          'rounds': [r], 'left_color': data['left_color'], 'right_color': data['right_color']})
                else:
                    if data['games'][-1]['rounds']:
                        r['round_number'] = data['games'][-1]['rounds'][-1]['round_number'] + 1
                    else:
                        r['round_number'] = 1
                    data['games'][-1]['rounds'].append(r)
                    if 'map' not in r:
                        r['map'] = 'oasis'

                    data['games'][-1]['map'] = r['map']
        print(data)
        print_game_data(data)
        #annotate_names(v, data)
        #annotate_statuses(v, data)
        #error
        res = upload_annotated_in_out_game(data)
        print(v)
        print(res)

        #error

        if not res['success']:
            error


def vod_main():
    #test()
    #error
    vods = get_annotate_vods_in_out_game()
    import subprocess
    actual_vods = []
    for v in vods:
        local_path = get_vod_path(v)
        if not os.path.exists(local_path):
            get_local_vod(v)
        if os.path.exists(local_path):
            check_vod_resolution(v)
            actual_vods.append(v)

    #error
    analyze_ingames(actual_vods)


if __name__ == '__main__':


    print('loaded model')
    # main()
    vod_main()
