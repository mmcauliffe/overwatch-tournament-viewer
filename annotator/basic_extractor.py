import os
import cv2
import json
import datetime
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, base_dir)

from .settings import *

from .classes import parse_match

sections = ['waiting', 'intro', 'roster', 'standings', 'analysts', 'casters', 'map_overview', 'in_game', 'paused',
            'score']

patches = []

data_dir = r'E:\Data\Overwatch\raw_data'
annotations_dir = os.path.join(data_dir, 'annotations')



def get_vod(match):
    match_dir = os.path.join(annotations_dir, match)
    vod_path = os.path.join(match_dir, '{}.mp4'.format(match))
    if os.path.exists(vod_path):
        return vod_path



def extract_segments(vod_path, match, games):
    match_dir = os.path.join(annotations_dir, match)
    print(match_dir)
    for g in games:
        print(g)
        game_dir = os.path.join(match_dir, g.id)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        for r in g.rounds:
            cap = cv2.VideoCapture(vod_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 35:
                reduce_amount = 6
            else:
                reduce_amount = 3
            begin_frame = int(r.begin * fps)
            end_frame = int(r.end * fps)
            if end_frame > num_frames:
                continue
            round_dir = os.path.join(game_dir, str(r.id))
            paths = {('left', x): os.path.join(round_dir, 'left_{}.avi'.format(x+1)) for x in range(6)}
            paths.update({('right', x): os.path.join(round_dir, 'right_{}.avi'.format(x+1)) for x in range(6)})
            exist_check = {k: os.path.exists(v) for k, v in paths.items()}
            writers = {k: cv2.VideoWriter(v, fourcc, fps / reduce_amount,
                                          (player_box_width, player_box_height)) for k, v in paths.items() if
                       not exist_check[k]}

            kill_feed_path = os.path.join(round_dir, 'kf.avi')
            if not writers and os.path.exists(kill_feed_path):
                continue
            kf_out = cv2.VideoWriter(kill_feed_path, fourcc, fps / reduce_amount,
                                     (kill_feed_box_width, kill_feed_box_total_height))
            cap.set(1, begin_frame)
            print(begin_frame, r.begin, num_frames, fps / reduce_amount)
            count = 0
            while True:
                ret, frame = cap.read()
                if count % reduce_amount == 0:
                    for k, v in writers.items():
                        v.write(box(frame, k[0], k[1], style=g.style))

                    kf_box = frame[kill_feed_box_y: kill_feed_box_y + kill_feed_box_total_height,
                             kill_feed_box_x: kill_feed_box_x + kill_feed_box_width]
                    kf_out.write(kf_box)
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                    break
                count += 1
            cap.release()
            for v in writers.values():
                v.release()

            kf_out.release()
            cv2.destroyAllWindows()

def extract_boxes(vod_path, match):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    match_dir = os.path.join(annotations_dir, match)
    cap = cv2.VideoCapture(vod_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 35:
        reduce_amount = 6
    else:
        reduce_amount = 3
    mid_path = os.path.join(match_dir, 'mid.avi')
    do_mid = not os.path.exists(mid_path)
    if not do_mid:
        return
    if do_mid:
        mid_out = cv2.VideoWriter(mid_path, fourcc, fps / reduce_amount, (mid_box_width, mid_box_height))
    count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        if do_mid and count % reduce_amount == 0:
            mid_box = frame[mid_box_y: mid_box_y + mid_box_height, mid_box_x: mid_box_x + mid_box_width]
            mid_out.write(mid_box)
        count += 1
    cap.release()
    if do_mid:
        mid_out.release()
    cv2.destroyAllWindows()


def find_offset(m, actual_start):
    path = os.path.join(annotations_dir, m, '1', '1_1_data.txt')
    with open(path, 'r', encoding='utf8') as f:
        data = json.load(f)
    wl_start = int(data['events'][0][0])
    return wl_start - actual_start


def train():
    matches = os.listdir(annotations_dir)
    for m in matches:
        if m in unusable:
            continue
        if m in ignored:
            continue
        match_dir = os.path.join(annotations_dir, m)
        if not os.path.exists(os.path.join(match_dir, '1')):
            continue
        vod_path = get_vod(m)
        if vod_path is None:
            continue
        print(m)
        games = parse_match(m)
        extract_segments(vod_path, m, games)
        extract_boxes(vod_path, m)


if __name__ == '__main__':
    train()
