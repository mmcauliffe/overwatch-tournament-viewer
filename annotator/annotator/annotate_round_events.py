import os
import json
import cv2
import numpy as np
import re
import torch
import itertools
import time
from collections import defaultdict, Counter
from annotator.config import offsets
from annotator.annotator.classes import MidAnnotator, PlayerStatusAnnotator, \
    KillFeedAnnotator, PlayerRNNAnnotator

from annotator.utils import get_local_vod, get_duration, \
    get_local_path,  FileVideoStream, Empty, get_vod_path
from annotator.api_requests import get_annotate_vods_round_events, upload_annotated_round_events, get_team

working_dir = r'N:\Data\Overwatch\models'
player_rnn_dir = os.path.join(working_dir, 'player_rnn')
player_model_dir = os.path.join(working_dir, 'player_status_test')
player_ocr_model_dir = os.path.join(working_dir, 'player_ocr')
kf_ctc_model_dir = os.path.join(working_dir, 'kill_feed_ctc_test')
kf_exists_model_dir = os.path.join(working_dir, 'kill_feed_exists')
game_model_dir = os.path.join(working_dir, 'game')
mid_model_dir = os.path.join(working_dir, 'mid')
replay_model_dir = os.path.join(working_dir, 'replay')
pause_model_dir = os.path.join(working_dir, 'pause')

annotation_dir = r'N:\Data\Overwatch\annotations'
oi_annotation_dir = r'N:\Data\Overwatch\oi_annotations'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
spec_modes = {'O': 'Original',
        'W':"world cup",
        "L":"overwatch league",
              'C': 'Contenders'}

sides = {'N': 'neither',
         'L': 'left',
         'R': 'right'}

use_status_rnn = False
debug = False

def predict_on_video(v, r):
    import time
    player_names = {}
    player_mapping = {}

    team = get_team(r['game']['left_team']['team'])
    print(team)
    for p in team['players']:
        player_mapping[p['id']] = p['name']
    team = get_team(r['game']['right_team']['team'])
    for p in team['players']:
        player_mapping[p['id']] = p['name']
    print(r['game']['right_team'])
    for p in r['game']['left_team']['players']:
        player_names[('left', p['player_index'])] = player_mapping[p['player']]
    for p in r['game']['right_team']['players']:
        player_names[('right', p['player_index'])] = player_mapping[p['player']]
    left_color = r['game']['left_team']['color'].lower()
    right_color = r['game']['right_team']['color'].lower()
    left_color_hex = r['game']['left_team_color_hex'].lstrip('#')
    right_color_hex = r['game']['right_team_color_hex'].lstrip('#')
    left_color_hex = tuple(int(left_color_hex[i:i+2], 16) for i in (0, 2, 4))
    right_color_hex = tuple(int(right_color_hex[i:i+2], 16) for i in (0, 2, 4))
    spec = r['game']['match']['event']['spectator_mode']['name'].lower()
    map = r['game']['map']['name'].lower()
    attacking_side = sides[r['attacking_side']]
    print(v['film_format'])
    film_format = v['film_format']['code']
    if use_status_rnn:
        status_annotator = PlayerRNNAnnotator(film_format, player_rnn_dir, device, left_color, right_color, player_names, spectator_mode=spec)
    else:
        status_annotator = PlayerStatusAnnotator(film_format, player_model_dir, device, left_color_hex, right_color_hex, player_names, spectator_mode=spec, debug=debug)
    status_annotator.get_zooms(r)
    #name_annotator = PlayerNameAnnotator(v['film_format'], player_ocr_model_dir, device)
    kill_feed_annotator = KillFeedAnnotator(film_format, kf_ctc_model_dir, kf_exists_model_dir, device,
                                            left_color_hex, right_color_hex,spectator_mode=spec, debug=debug)
    mid_annotator = MidAnnotator(film_format, mid_model_dir, device, spec, map, attacking_side, debug=debug)

    actual_duration, mode = get_duration(r['stream_vod'])

    offset = None
    if r['stream_vod']['id'] in offsets:
        offset = offsets[r['stream_vod']['id']]
    for beg, end in r['sequences']:
        print(beg, end)
        if end <= beg:
            continue
        time_step = 0.1
        fvs = FileVideoStream(get_vod_path(v), beg + r['begin'], end + r['begin'], time_step, real_begin=r['begin'],
                              offset=offset, actual_duration=actual_duration, mode=mode).start()
        time.sleep(5)
        frame_ind = 0
        status_annotator.reset(beg)
        mid_annotator.reset(beg)
        while True:
            try:
                data, time_point = fvs.read()
            except Empty:
                break
            frame = data['frame']
            if time_point % (status_annotator.time_step * status_annotator.batch_size) == 0:
                print(time_point)

            if time_step == status_annotator.time_step or frame_ind % (status_annotator.time_step / time_step) == 0:
                status_annotator.process_frame(frame, time_point)
            kill_feed_annotator.process_frame(frame, time_point)
            #name_annotator.process_frame(frame, time_point)
            mid_annotator.process_frame(frame, time_point)
            frame_ind += 1
        status_annotator.annotate()
        #name_annotator.annotate()
        mid_annotator.annotate()
    mid_statuses = mid_annotator.generate_round_properties(r)
    #print(name_annotator.names)
    #statuses = status_annotator.generate_statuses()
    left_team, right_team = status_annotator.generate_teams()
    kill_feed_events = kill_feed_annotator.generate_kill_events(left_team, right_team, round(r['end'] - r['begin'], 1))

    print(kill_feed_events)
    if False:
        for s, status in statuses.items():
            print(s)
            for k, intervals in status.items():
                print('   ', k)
                for interval in intervals:
                    begin_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['begin'] + r['begin']))
                    end_timestamp = time.strftime('%H:%M:%S', time.gmtime(interval['end'] + r['begin']))
                    print('      ', '{}-{}: {}'.format(begin_timestamp, end_timestamp, interval['status']))
    data_player_states = {}
    for t in [left_team, right_team]:
        side = t.side
        enemy_team = left_team
        if t == left_team:
            enemy_team = right_team
        for k, v in t.player_states.items():
            k = '{}_{}'.format(side, k)
            print(k)
            data_player_states[k] = {}
            switches = v.generate_switches()
            ultimates = v.generate_ults(mech_deaths=kill_feed_annotator.mech_deaths, revive_events=kill_feed_annotator.dva_revive_events, sequences=r['sequences'])
            data_player_states[k]['player_name'] = v.name
            data_player_states[k]['switches'] = switches
            data_player_states[k]['ultimates'] = ultimates
            # status effects
            data_player_states[k].update(v.generate_status_effects(enemy_team=enemy_team, friendly_team=t))
    #min_time = None
    #for time_point, count in list(ult_gain_counts.items()) + list(ult_use_counts.items()):
    #    if count > 5:
    #        if min_time is None or time_point < min_time:
    #            min_time = time_point
    #if min_time is not None:
    #    for t in [left_team, right_team]:
    #        side = t.side
    #        for k, v in t.player_states.items():
    #            k = '{}_{}'.format(side, k)
    #            data_player_states[k]['ult_gains'] = [x for x in data_player_states[k]['ult_gains'] if x < min_time]
    #            data_player_states[k]['ult_uses'] = [x for x in data_player_states[k]['ult_uses'] if x < min_time]
    statuses = {'player': data_player_states, 'kill_feed': kill_feed_events, 'left_color': left_team.color,
            'right_color': right_team.color}
    statuses.update(mid_statuses)
    for k,v in statuses.items():
        print(k)
        print(v)
    return statuses


def analyze_rounds(vods):
    import random
    #random.seed(1234)
    game_dir = os.path.join(oi_annotation_dir, 'to_check')
    annotation_dir = os.path.join(oi_annotation_dir, 'annotations')
    random.shuffle(vods)
    for i, v in enumerate(vods):
        #if "VAL" not in v['title']:
        #    continue
        print(i, v)
        random.shuffle(v['rounds'])
        for r in v['rounds']:
            #if r['spectator_mode'].lower() != 'overwatch league season 3':
            #    continue
            #if r['id'] < 15025:
            #    continue
            #if r['id'] != 15311:
            #    continue
            print(r)
            print(r['begin'], r['end'])
            print(r['sequences'])
            #replays, pauses, smaller_windows, sequences = get_sequences(v, r)
            replays, pauses, smaller_windows = [], [], []
            #print('r', replays)
            #print('p', pauses)
            #print('s', sequences)
            #r['sequences'] = [(100, 120)] # FIXME
            #round_props = get_round_status(get_vod_path(v), r['begin'], r['end'], v['film_format'], sequences)

            data = predict_on_video(v, r)
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
            print('DATA')
            print(data)
            #error
            resp = upload_annotated_round_events(data)

            print(resp)
            if not resp['success']:
                error
            #error
            #break

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
