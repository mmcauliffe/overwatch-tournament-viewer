import os
import cv2
import json

sections = ['waiting', 'intro', 'roster', 'standings', 'analysts', 'casters', 'map_overview', 'in_game', 'paused',
            'score']

heroes = ['doomfist', 'genji', 'mccree', 'pharah', 'reaper', 'soldier: 76', 'sombra', 'tracer',
          'bastion', 'hanzo', 'junkrat', 'mei', 'torbjorn', 'widowmaker',
          'd.va', 'orisa', 'reinhardt', 'roadhog', 'winston', 'zarya',
          'ana', 'lucio', 'mercy', 'symmetra', 'zenyatta']

maps = ['hanamura', 'temple of anubis', 'volskaya industries',
        'dorado', 'route 66', 'watchpoint: gibraltar',
        'ilios', 'lijiang tower', 'nepal', 'oasis',
        'eichenwalde', 'hollywood', "king's row", 'numbani']

patches = []

data_dir = r'D:\Data\Overwatch\raw_data'
annotations_dir = os.path.join(data_dir, 'annotations')


def get_vod(match, meta):
    match_dir = os.path.join(annotations_dir, match)
    vod_path = os.path.join(match_dir, '{}.mp4'.format(match))
    if os.path.exists(vod_path):
        return vod_path


player_box_y = 45
blue_box_x = 30
red_box_x = 832
player_box_height = 55
player_box_width = 67
blue_box_width = 420
margin = 4

mid_box_y = player_box_y
mid_box_x = 490
mid_box_width = 300
mid_box_height = 140

kill_feed_box_y = 115
kill_feed_box_x = 950
kill_feed_box_width = 300
kill_feed_box_total_height = 205
kill_feed_box_height = int(kill_feed_box_total_height / 6)


def box(frame, team, number):
    if team == 'blue':
        x = blue_box_x
    else:
        x = red_box_x
    if number != 1:
        x += (player_box_width + margin) * (number - 1)
    return frame[player_box_y: player_box_y + player_box_height, x: x + player_box_width]


def kf_box(frame, team, number):
    y = kill_feed_box_y
    if number != 1:
        y += (kill_feed_box_height) * (number - 1)
    return frame[y: y + kill_feed_box_height, kill_feed_box_x: kill_feed_box_x + kill_feed_box_width]

def extract_segments(vod_path, match, meta):
    match_dir = os.path.join(annotations_dir, match)
    print(match_dir)
    games = [x for x in os.listdir(match_dir) if os.path.isdir(os.path.join(match_dir, x))]
    print(games)
    for g in games:
        print(g)
        game_dir = os.path.join(match_dir, g)
        meta_path = os.path.join(game_dir, 'meta.json')
        with open(meta_path, 'r', encoding='utf8') as f:
            game_meta = json.load(f)
            print(game_meta)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        begins, ends = game_meta['begins'], game_meta['ends']
        num_rounds = len(begins)
        for i in range(1, num_rounds+1):
            begin = begins[i-1]
            end = ends[i-1]
            cap = cv2.VideoCapture(vod_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            begin_frame = int(begin * fps)
            end_frame = int(end * fps)
            paths = {('blue', x): os.path.join(game_dir, '{}_{}_blue_{}.avi'.format(g, i, x)) for x in range(1, 7)}
            paths.update({('red', x): os.path.join(game_dir, '{}_{}_red_{}.avi'.format(g, i, x)) for x in range(1, 7)})
            paths.update({('kf', x): os.path.join(game_dir, '{}_{}_kf_{}.avi'.format(g, i, x)) for x in range(1, 7)})
            exist_check = {k: os.path.exists(v) for k, v in paths.items()}
            writers = {k: cv2.VideoWriter(v, fourcc, fps,
                                          (player_box_width, player_box_height)) for k, v in paths.items() if
                       not exist_check[k] and k[0] != 'kf'}
            #kf_writers = {k: cv2.VideoWriter(v, fourcc, fps,
            #                              (kill_feed_box_width, kill_feed_box_height)) for k, v in paths.items() if
            #           not exist_check[k] and k[0] == 'kf'}

            kill_feed_path = os.path.join(game_dir, '{}_{}_kf.avi'.format(g, i))
            if not writers and os.path.exists(kill_feed_path):
                continue
            kf_out = cv2.VideoWriter(kill_feed_path, fourcc, fps,
                                     (kill_feed_box_width, kill_feed_box_total_height))
            cap.set(1, begin_frame)
            while True:
                ret, frame = cap.read()
                for k, v in writers.items():
                    v.write(box(frame, k[0], k[1]))
                #for k, v in kf_writers.items():
                #    v.write(kf_box(frame, k[0], k[1]))
                kf_box = frame[kill_feed_box_y: kill_feed_box_y + kill_feed_box_total_height,
                         kill_feed_box_x: kill_feed_box_x + kill_feed_box_width]
                kf_out.write(kf_box)
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                    break
            cap.release()
            for v in writers.values():
                v.release()
            #for v in kf_writers.values():
            #    v.release()
            kf_out.release()
            cv2.destroyAllWindows()


def extract_boxes(vod_path, match, meta):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    match_dir = os.path.join(annotations_dir, match)
    cap = cv2.VideoCapture(vod_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    mid_path = os.path.join(match_dir, 'mid.avi')
    do_mid = not os.path.exists(mid_path)
    if not do_mid:
        return
    if do_mid:
        mid_out = cv2.VideoWriter(mid_path, fourcc, fps, (mid_box_width, mid_box_height))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        if do_mid:
            mid_box = frame[mid_box_y: mid_box_y + mid_box_height, mid_box_x: mid_box_x + mid_box_width]
            mid_out.write(mid_box)
    cap.release()
    if do_mid:
        mid_out.release()
    cv2.destroyAllWindows()


def generate_events(match, meta):
    match_dir = os.path.join(annotations_dir, match)
    print(match_dir)
    games = [x for x in os.listdir(match_dir) if os.path.isdir(os.path.join(match_dir, x))]
    print(games)
    for g in games:
        print(g)
        game_dir = os.path.join(match_dir, g)
        meta_path = os.path.join(game_dir, 'meta.json')
        with open(meta_path, 'r', encoding='utf8') as f:
            game_meta = json.load(f)
            print(game_meta)
        for r in range(1, 10):
            events_path = os.path.join(game_dir, '{}_{}_data.txt'.format(g, r))
            if not os.path.exists(events_path):
                break
            with open(events_path, 'r', encoding='utf8') as f:
                events = json.load(f)['events']
            print(r)
            match_begin = events[0][0]
            player_heroes = {('blue', str(x)): [] for x in range(1,7)}
            player_heroes.update({('red', str(x)): [] for x in range(1,7)})
            ult_status = {('blue', str(x)): [] for x in range(1,7)}
            ult_status.update({('red', str(x)): [] for x in range(1,7)})
            deaths = {('blue', str(x)): [] for x in range(1,7)}
            deaths.update({('red', str(x)): [] for x in range(1,7)})
            kill_feed = []
            for e in events:
                print(e)
                time = int(e[0])
                event_type = e[1]
                if event_type in ['MATCH', 'ATTACK', 'END', 'POINTS', 'PAUSE', 'UNPAUSE']:
                    if event_type == 'MATCH':
                        pass
                    pass
                elif event_type in ['SWITCH', 'DEATH', 'ULT_GAIN', 'ULT_USE', 'KILL']:
                    time -= match_begin
                    team = e[2].lower()
                    player = e[3]

                    if event_type == 'SWITCH':
                        new_hero = e[-1]
                        player_heroes[(team, player)].append((time, new_hero))
                    elif event_type == 'ULT_GAIN':
                        ult_status[(team, player)].append((time, 'ult_gain'))
                    elif event_type == 'ULT_USE':
                        ult_status[(team, player)].append((time, 'ult_use'))
                    elif event_type == 'DEATH':
                        if player != '0':
                            deaths[(team, player)].append(time)

                        character = e[4]
                        kill_feed.append((time, 'None', 'None', 'regular', team, character))
                    elif event_type == 'KILL':
                        character = e[4]
                        killed_character = e[6]
                        method = e[7] if e[7] else 'regular'
                        killed_team = 'blue' if team == 'red' else 'red'
                        kill_feed.append((time, team, character, method, killed_team, killed_character))
                        if team == 'blue':
                            team = 'red'
                        else:
                            team = 'blue'
                        player = e[5]
                        if player != '0':
                            deaths[(team, player)].append(time)
                else:
                    print(event_type)
                    error
            for k, v in player_heroes.items():
                with open(os.path.join(game_dir, '{}_{}_{}_{}_heroes.txt'.format(g, r, k[0], k[1])), 'w', encoding='utf') as f:
                    for line in v:
                        f.write('{} {}\n'.format(*line))
            for k, v in ult_status.items():
                with open(os.path.join(game_dir, '{}_{}_{}_{}_ults.txt'.format(g, r, k[0], k[1])), 'w', encoding='utf') as f:
                    for line in v:
                        f.write('{} {}\n'.format(*line))
            for k, v in deaths.items():
                with open(os.path.join(game_dir, '{}_{}_{}_{}_deaths.txt'.format(g, r, k[0], k[1])), 'w', encoding='utf') as f:
                    for line in v:
                        f.write('{}\n'.format(line))
            with open(os.path.join(game_dir, '{}_{}_kf.txt'.format(g, r)), 'w', encoding='utf') as f:
                for i, line in enumerate(kill_feed):
                    if line[1] == 'None':
                        same_times = [x for x in kill_feed if abs(x[0] - line[0]) <= 5]
                        found_kill = False
                        for s in same_times:
                            if line[4] == s[4] and line[5] == s[5] and s[1] != 'None':
                                found_kill = True
                        if found_kill:
                            continue
                    f.write('{}\n'.format(' '.join(map(str, line))))
            #error


def train():
    matches = os.listdir(annotations_dir)
    for m in matches:
        if m != '513':
            continue
        match_dir = os.path.join(annotations_dir, m)
        if not os.path.exists(os.path.join(match_dir, '1')):
            continue
        meta_path = os.path.join(match_dir, 'metadata.json')
        with open(meta_path, 'r', encoding='utf8') as f:
            meta = json.load(f)
            print(meta)
        vod_path = get_vod(m, meta)
        extract_segments(vod_path, m, meta)
        extract_boxes(vod_path, m, meta)
        generate_events(m, meta)


def annotate_file(to_annotate):
    cap = cv2.VideoCapture(to_annotate)
    ret, frame = cap.read()
    position = 210626

    while True:
        if frame is None:
            break
        cv2.imshow('frame', frame)
        k = cv2.waitKey(33)
        if k == 27:
            break
        elif k == 255:
            continue
        elif k in [37, 38, 39, 40]:
            if k == 37:
                position -= 1
            elif k == 39:
                position += 1
            elif k == 38:
                position += 60
            elif k == 40:
                position -= 60
            if position < 0:
                position = 0
            print(position, position / 60)
            cap.set(1, position)
            ret, frame = cap.read()
        else:
            print(k)


if __name__ == '__main__':
    train()
