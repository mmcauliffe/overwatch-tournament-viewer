import os
import json
import cv2
import numpy as np
import re
import torch
import itertools
import time
from collections import defaultdict, Counter
from annotator.annotator.classes import InGameAnnotator, MidAnnotator, PlayerNameAnnotator, PlayerStatusAnnotator, KillFeedAnnotator
from annotator.annotator.classes.base import filter_statuses

from annotator.utils import get_local_vod, \
    get_local_path,  FileVideoStream, Empty, get_vod_path
from annotator.api_requests import get_annotate_vods_in_out_game, upload_annotated_in_out_game

frames_per_seq = 100

working_dir = r'E:\Data\Overwatch\models'
player_model_dir = os.path.join(working_dir, 'player_status')
player_ocr_model_dir = os.path.join(working_dir, 'player_ocr')
kf_ctc_model_dir = os.path.join(working_dir, 'kill_feed_ctc')
game_model_dir = os.path.join(working_dir, 'game')
mid_model_dir = os.path.join(working_dir, 'mid')

annotation_dir = r'E:\Data\Overwatch\annotations'
oi_annotation_dir = r'E:\Data\Overwatch\oi_annotations'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def extract_info(v):
    owl_mapping = {'DAL': 'Dallas Fuel',
                   'PHI': 'Philadelphia Fusion', 'SEO': 'Seoul Dynasty',
                   'LDN': 'London Spitfire', 'SFS': 'San Francisco Shock', 'HOU': 'Houston Outlaws',
                   'BOS': 'Boston Uprising', 'VAL': 'Los Angeles Valiant', 'GLA': 'Los Angeles Gladiators',
                   'FLA': 'Florida Mayhem', 'SHD': 'Shanghai Dragons', 'NYE': 'New York Excelsior',
                   'PAR': 'Paris Eternal', 'TOR': 'Toronto Defiant', 'WAS': 'Washington Justice',
                   'VAN': 'Vancouver Titans', 'CDH': 'Chengdu Hunters', 'HZS': 'Hangzhou Spark',
                   'ATL': 'Atlanta Reign', 'GZC': 'Guangzhou Charge'}
    channel = v['channel']['name']
    info = {}
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
    elif channel.lower() == 'overwatchleague':
        pattern = r'Game (\d+) (\w+) @ (\w+) \| ([\w ]+)'
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

    return info


def annotate_game_or_not(v):
    game_annotator = InGameAnnotator(v['film_format'], game_model_dir, device)
    fvs = FileVideoStream(get_vod_path(v), 0, 0, game_annotator.time_step).start()
    time.sleep(5)
    print('begin ingame/outofgame processing')
    while True:
        try:
            frame, time_point = fvs.read()
        except Empty:
            break
        game_annotator.process_frame(frame)
    game_annotator.annotate()
    return game_annotator.status


def annotate_names(v, data):
    import time
    for r in data['rounds']:
        print('begin', time.strftime('%H:%M:%S', time.gmtime(r['begin'])))
        print('end', time.strftime('%H:%M:%S', time.gmtime(r['end'])))
        name_annotator = PlayerNameAnnotator(v['film_format'], player_ocr_model_dir, device)
        fvs = FileVideoStream(get_vod_path(v), r['begin'], r['end'], name_annotator.time_step, real_begin=r['begin']).start()
        time.sleep(5)
        print('begin name processing')
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            name_annotator.process_frame(frame)
        name_annotator.annotate()
        names = name_annotator.generate_names()
        for s, name in names.items():
            print(s, name)


def annotate_statuses(v, data):
    import time
    for r in data['rounds']:
        print('begin', time.strftime('%H:%M:%S', time.gmtime(r['begin'])))
        print('end', time.strftime('%H:%M:%S', time.gmtime(r['end'])))
        status_annotator = PlayerStatusAnnotator(v['film_format'], player_model_dir, device)
        print('ONLY DOING 60 seconds')
        fvs = FileVideoStream(get_vod_path(v), r['begin'], r['begin']+200, status_annotator.time_step, real_begin=r['begin']).start()
        time.sleep(5)
        print('begin status processing')
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            print(time_point, r['end'] - r['begin'])
            status_annotator.process_frame(frame)
        status_annotator.annotate()
        statuses = status_annotator.generate_statuses()
        for s, status in statuses.items():
            print(s)
            for k, intervals in status.items():
                print('   ', k)
                for interval in intervals:
                    begin_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['begin'] + r['begin']))
                    end_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['end'] + r['begin']))
                    print('      ', '{}-{}: {}'.format(begin_timestamp, end_timestamp, interval['status']))
        error


def annotate_properties(v, data):
    import time
    round_level = ['round_number', 'attacking_color']
    game_level = ['spectator_mode', 'map', 'map_mode']
    for r in data['rounds']:
        print('begin', time.strftime('%H:%M:%S', time.gmtime(r['begin'])))
        print('end', time.strftime('%H:%M:%S', time.gmtime(r['end'])))
        mid_annotator = MidAnnotator(v['film_format'], mid_model_dir, device)
        fvs = FileVideoStream(get_vod_path(v), r['begin'], r['end'], mid_annotator.time_step, real_begin=r['begin']).start()
        time.sleep(5)
        print('begin mid processing')
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            mid_annotator.process_frame(frame)
        mid_annotator.annotate()
        statuses = mid_annotator.statuses
        ends = {}
        for k in round_level:
            counter = defaultdict(float)
            for i in statuses[k]:
                counter[i['status']] += i['end'] - i['begin']
            print(counter)
            ends[k] = max(counter.keys(), key=lambda x: counter[x])


def analyze_ingames(vods):
    game_dir = os.path.join(oi_annotation_dir, 'to_check')
    os.makedirs(game_dir, exist_ok=True)
    for v in vods:
        info= extract_info(v)
        if not info:
            continue
        data = {'vod_id': v['id'], 'rounds': [], 'team_one': info['team_one'].lower(), 'team_two': info['team_two'].lower()}
        if v['type'] == 'G' and 'game_number' in info:
            data['game_number'] = int(info['game_number'])
        print(data)
        game_status = annotate_game_or_not(v)
        print(game_status)
        for interval in game_status:
            begin_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['begin']))
            end_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['end']))
            print('      ', '{}-{}: {}'.format(begin_timestamp, end_timestamp, interval['status']))
        game_status = filter_statuses(game_status, 20)

        for g in game_status:
            if g['status'] == 'game':
                begin_timestamp = time.strftime('%H:%M:%S', time.gmtime(g['begin']))
                end_timestamp = time.strftime('%H:%M:%S', time.gmtime(g['end']))
                print('GAME      ', '{}-{}: {}'.format(begin_timestamp, end_timestamp, g['status']))
                data['rounds'].append({'begin': g['begin'], 'end': g['end']})

        print(data)
        #annotate_names(v, data)
        #annotate_statuses(v, data)
        print(upload_annotated_in_out_game(data))


def vod_main():
    #test()
    #error
    vods = get_annotate_vods_in_out_game()
    for v in vods:
        local_path = get_vod_path(v)
        if not os.path.exists(local_path):
            get_local_vod(v)

    analyze_ingames(vods)


if __name__ == '__main__':


    print('loaded model')
    # main()
    vod_main()
