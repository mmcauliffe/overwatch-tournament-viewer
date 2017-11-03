import cv2
import os
import numpy as np
from PIL import Image
import pytesseract
import subprocess

base_dir = r'D:\Data\Overwatch'
data_dir = r'D:\Data\Overwatch\Carbon_series'
ref_dir = os.path.join(base_dir, 'reference')
ult_dir = os.path.join(ref_dir, 'ult')


def load_ult_digits():
    digits = list('01234567890')
    ref = {}
    for d in digits:
        d_path = os.path.join(ult_dir, 'ult_{}.png'.format(d))
        if os.path.exists(d_path):
            ref[d] = cv2.imread(d_path)
    return ref


file_path = os.path.join(data_dir, 'OCS_03132017.mp4')

ult_box_height = 18
ult_box_width = 24
player_box_height = 54
player_box_width = 67
player_name_box_height = 16
player_name_box_width = 67
player_box_y = 45
ult_y = player_box_y + 10
name_y = player_box_y + 30
blue_start = 29
ult_blue_start = blue_start + 2
offset = 70.6
red_start = 831
ult_red_start = red_start + 3
player_boxes = [(player_box_y, blue_start), (player_box_y, blue_start + int(offset)),
                (player_box_y, blue_start + int(offset * 2)), (player_box_y, blue_start + int(offset * 3)),
                (player_box_y, blue_start + int(offset * 4)), (player_box_y, blue_start + int(offset * 5)),

                (player_box_y, red_start), (player_box_y, red_start + int(offset)),
                (player_box_y, red_start + int(offset * 2)), (player_box_y, red_start + int(offset * 3)),
                (player_box_y, red_start + int(offset * 4)), (player_box_y, red_start + int(offset * 5))]

ult_boxes = [(ult_y, ult_blue_start), (ult_y, ult_blue_start + int(offset)),
             (ult_y, ult_blue_start + int(offset * 2)), (ult_y, ult_blue_start + int(offset * 3)),
             (ult_y, ult_blue_start + int(offset * 4)), (ult_y, ult_blue_start + int(offset * 5)),

             (ult_y, ult_red_start), (ult_y, ult_red_start + int(offset)), (ult_y, ult_red_start + int(offset * 2)),
             (ult_y, ult_red_start + int(offset * 3)), (ult_y, ult_red_start + int(offset * 4)),
             (ult_y, ult_red_start + int(offset * 5))]

player_name_boxes = [(name_y, blue_start), (name_y, blue_start + int(offset)),
                     (name_y, blue_start + int(offset * 2)), (name_y, blue_start + int(offset * 3)),
                     (name_y, blue_start + int(offset * 4)), (name_y, blue_start + int(offset * 5)),

                     (name_y, red_start), (name_y, red_start + int(offset)), (name_y, red_start + int(offset * 2)),
                     (name_y, red_start + int(offset * 3)), (name_y, red_start + int(offset * 4)),
                     (name_y, red_start + int(offset * 5))]

teams = ['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'red', 'red', 'red', 'red', 'red', 'red']
heroes = ['lucio', 'ana', 'soldier', 'tracer', 'genji', 'winston',
          'tracer', 'zarya', 'winston', 'ana', 'genji', 'lucio']


def player_ult_box(image, index):
    corner = ult_boxes[index]
    return image[corner[0]:corner[0] + ult_box_height,
           corner[1]: corner[1] + ult_box_width]


def player_name_box(image, index):
    corner = player_name_boxes[index]
    return image[corner[0]:corner[0] + player_name_box_height,
           corner[1]: corner[1] + player_name_box_width]


def player_box(image, index):
    corner = player_boxes[index]
    return image[corner[0]:corner[0] + player_box_height,
           corner[1]: corner[1] + player_box_width]


# thresholds for extracting numbers for each team + character combo

thresholds = {('blue', 'lucio'): 175,
              ('blue', 'ana'): 192,
              ('blue', 'soldier'): 197,
              ('blue', 'tracer'): 192,
              ('blue', 'genji'): 180,
              ('blue', 'winston'): 195,
              ('red', 'lucio'): 192,
              ('red', 'ana'): 192,
              ('red', 'zarya'): 197,
              ('red', 'tracer'): 192,
              ('red', 'genji'): 192,
              ('red', 'winston'): 192,
              }


def generate_train_data():
    frames_to_read = 10
    frame_count = 0
    cap = cv2.VideoCapture(file_path)
    cap.set(1, 150000)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))

    to_cat = 1
    train = np.empty((0, 10 * ult_box_width))
    row = np.zeros((ult_box_height, 10 * ult_box_width))
    row[:, :] = 255
    while frame_count < frames_to_read:

        ret, frame = cap.read()
        cap.set(1, 150000 + frame_count * 120)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_count += 1
        col_count = 0

        for i, ub in enumerate(ult_boxes):
            print(i, teams[i], heroes[i])
            gray = cv2.cvtColor(player_ult_box(frame, i), cv2.COLOR_BGR2GRAY)
            # print(gray)
            ref = cv2.threshold(gray, thresholds[(teams[i], heroes[i])], 255, cv2.THRESH_BINARY_INV)[1]
            # print(ref)
            print(ref.shape)
            # if i in [6,9]:
            #    cv2.imwrite(os.path.join(ult_dir, 'ult_ready.png'.format(to_cat)),ref)
            #    continue
            cv2.imshow('reference', gray)
            cv2.imshow('frame', ref)
            key = cv2.waitKey(0)
            if key == ord('n'):
                ref[:5, :] = 255
            elif key == ord('s'):
                continue
            row[:, ult_box_width * col_count:ult_box_width * (col_count + 1)] = ref
            col_count += 1
            if col_count == 10:
                col_count = 0
                print(train.shape, row.shape)
                train = np.vstack((train, row))
                print(train.shape)
                row = np.zeros((ult_box_height, 10 * ult_box_width))
                row[:, :] = 255

            first = ref[:, :8]
            second = ref[:, 7:]
            # print(first.shape, second.shape)
            cv2.imwrite(os.path.join(ult_dir, 'to_cat_{}.png'.format(to_cat)), first)
            # cv2.imshow('firstnumber_{}'.format(i), first)
            # cv2.imshow('secondnumber_{}'.format(i), second)
            to_cat += 1
            cv2.imwrite(os.path.join(ult_dir, 'to_cat_{}.png'.format(to_cat)), second)
            to_cat += 1
    cv2.imwrite(os.path.join(ult_dir, 'train.png'), train)
    print(to_cat)


def show_ult_areas():
    cap = cv2.VideoCapture(file_path)
    cap.set(1, 145000)
    ret, frame = cap.read()
    annotated = frame.copy()
    for i, ub in enumerate(ult_boxes):
        top_left = ub[1], ub[0]
        bottom_right = (ub[1] + ult_box_width, ub[0] + ult_box_height)
        print(top_left, bottom_right)
        cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 3)
        orig = player_ult_box(frame, i)
        text = image_to_string(orig)
        print(text)
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        # print(gray)
        ref = cv2.threshold(gray, thresholds[(teams[i], heroes[i])], 255, cv2.THRESH_BINARY_INV)[1]
        # ref = cv2.inRange(orig, (200, 170, 170), (255, 255, 255))
        # print(orig)
        # print(orig.shape)
        # error
        cv2.imshow('gray', gray)
        cv2.imshow('ref', ref)

        # cv2.imshow('norm', orig)
        # while True:
        #    ch = 0xFF & cv2.waitKey(1)  # Wait for a second
        #    if ch == 27:
        #        error
        cv2.imshow('{}_binary'.format(i), ref)
    cv2.imshow('frame', annotated)
    cv2.imshow('first', player_ult_box(frame, 0))
    cv2.imshow('second', player_ult_box(frame, 1))
    cv2.imshow('third', player_ult_box(frame, 2))
    cv2.imshow('fourth', player_ult_box(frame, 3))
    cv2.imshow('fifth', player_ult_box(frame, 4))
    cv2.imshow('sixth', player_ult_box(frame, 5))
    cv2.imshow('seventh', player_ult_box(frame, 6))
    cv2.imshow('eighth', player_ult_box(frame, 7))
    cv2.imshow('ninth', player_ult_box(frame, 8))
    cv2.imshow('tenth', player_ult_box(frame, 9))
    cv2.imshow('eleventh', player_ult_box(frame, 10))
    cv2.imshow('twelfth', player_ult_box(frame, 11))
    while True:
        ch = 0xFF & cv2.waitKey(1)  # Wait for a second
        if ch == 27:
            break


def show_player_areas():
    cap = cv2.VideoCapture(file_path)
    cap.set(1, 145000)
    ret, frame = cap.read()
    annotated = frame.copy()
    for i, ub in enumerate(player_boxes):
        top_left = ub[1], ub[0]
        bottom_right = (ub[1] + player_box_width, ub[0] + player_box_height)
        print(top_left, bottom_right)
        cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 3)
        orig = player_ult_box(frame, i)
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        # print(gray)
        ref = cv2.threshold(gray, thresholds[(teams[i], heroes[i])], 255, cv2.THRESH_BINARY_INV)[1]
        # ref = cv2.inRange(orig, (200, 170, 170), (255, 255, 255))
        # print(orig)
        # print(orig.shape)
        # error
        cv2.imshow('gray', gray)
        cv2.imshow('ref', ref)

        # cv2.imshow('norm', orig)
        # while True:
        #    ch = 0xFF & cv2.waitKey(1)  # Wait for a second
        #    if ch == 27:
        #        error
        cv2.imshow('{}_binary'.format(i), ref)
    cv2.imshow('frame', annotated)
    cv2.imshow('first', player_box(frame, 0))
    cv2.imshow('second', player_box(frame, 1))
    cv2.imshow('third', player_box(frame, 2))
    cv2.imshow('fourth', player_box(frame, 3))
    cv2.imshow('fifth', player_box(frame, 4))
    cv2.imshow('sixth', player_box(frame, 5))
    cv2.imshow('seventh', player_box(frame, 6))
    cv2.imshow('eighth', player_box(frame, 7))
    cv2.imshow('ninth', player_box(frame, 8))
    cv2.imshow('tenth', player_box(frame, 9))
    cv2.imshow('eleventh', player_box(frame, 10))
    cv2.imshow('twelfth', player_box(frame, 11))
    while True:
        ch = 0xFF & cv2.waitKey(1)  # Wait for a second
        if ch == 27:
            break


# cv2.imshow('frame', frame)
# while True:
#    ch = 0xFF & cv2.waitKey(1)  # Wait for a second
#    if ch == 27:
#        break

def image_to_string(img, word=False):
    height, width = img.shape[:2]
    orig = cv2.resize(img, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    if word:
        ref = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]
    else:
        ref = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    path = os.path.join(ult_dir, 't.png')
    cv2.imwrite(path, ref)
    player_names_path = r'D:/Data/Overwatch/player_names.txt'
    ult_percent_path = r'D:/Data/Overwatch/ult_percent.txt'
    if word:
        result = subprocess.run(
            ['tesseract', path, 'stdout', '-psm', '8', '-l', 'eng', '--user-words', player_names_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    else:
        result = subprocess.run(
            ['tesseract', path, 'stdout', 'digits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    print(result.stderr)
    text = result.stdout.decode('utf8').strip()
    return text


def show_name_areas():
    cap = cv2.VideoCapture(file_path)
    cap.set(1, 145000)
    ret, frame = cap.read()
    annotated = frame.copy()
    for i, ub in enumerate(player_name_boxes):
        top_left = ub[1], ub[0]
        bottom_right = (ub[1] + player_name_box_width, ub[0] + player_name_box_height)
        print(top_left, bottom_right)
        cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 3)
        orig = player_name_box(frame, i)
        height, width = orig.shape[:2]
        orig = cv2.resize(orig, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)
        text = image_to_string(orig)
    cv2.imshow('frame', annotated)
    # cv2.imshow('first', player_name_box(frame, 0))
    # cv2.imshow('second', player_name_box(frame, 1))
    # cv2.imshow('third', player_name_box(frame, 2))
    # cv2.imshow('fourth', player_name_box(frame, 3))
    # cv2.imshow('fifth', player_name_box(frame, 4))
    # cv2.imshow('sixth', player_name_box(frame, 5))
    # cv2.imshow('seventh', player_name_box(frame, 6))
    # cv2.imshow('eighth', player_name_box(frame, 7))
    # cv2.imshow('ninth', player_name_box(frame, 8))
    # cv2.imshow('tenth', player_name_box(frame, 9))
    # cv2.imshow('eleventh', player_name_box(frame, 10))
    # cv2.imshow('twelfth', player_name_box(frame, 11))
    # while True:
    #    ch = 0xFF & cv2.waitKey(1)  # Wait for a second
    #    if ch == 27:
    #        break


if __name__ == '__main__':
    # find_player_squares()
    # show_player_areas()
    show_ult_areas()
    show_name_areas()
    # generate_train_data()
