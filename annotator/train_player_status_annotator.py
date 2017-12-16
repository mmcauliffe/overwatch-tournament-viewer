import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
import numpy as np

from .settings import *
from .classes import *

ignored += ['2360', '2359', '2358', '2357', '2356', '2355']

working_dir = r'E:\Data\Overwatch\models'
os.makedirs(working_dir, exist_ok=True)

data_dir = r'E:\Data\Overwatch\raw_data'
train_dir = r'C:\Users\micha\Documents\Data\player_train'
annotations_dir = os.path.join(data_dir, 'annotations')

hero_label_file = os.path.join(train_dir, 'hero_labels.npy')
death_label_file = os.path.join(train_dir, 'death_labels.npy')
has_ult_label_file = os.path.join(train_dir, 'has_ult_labels.npy')


def generate_train_data():
    os.makedirs(train_dir, exist_ok=True)
    labels = []
    frame_id = 0
    for m in test_files:
        if m in ignored:
            continue
        print(m)
        games = parse_match(m)
        match_dir = os.path.join(annotations_dir, m)

        mid_file = os.path.join(match_dir, 'mid.avi')
        print(mid_file)
        cap = cv2.VideoCapture(mid_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        actual_start = None
        if m in actual_starts:
            actual_start = actual_starts[m]
        events = load_events(match_dir, fps, actual_start)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(num_frames)
        frame_count = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            label = get_label(frame_count, events)
            np.save(os.path.join(train_dir, '{}.npy'.format(frame_id)), frame)
            frame_count += 1
            frame_id += 1
            labels.append(cnn_output_categories.index(label))
        cap.release()
    np.save(label_file, np.array(labels))