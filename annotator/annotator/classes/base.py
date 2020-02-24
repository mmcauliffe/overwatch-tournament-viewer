import numpy as np
import cv2
from annotator.config import BOX_PARAMETERS


def coalesce_statuses(statuses):
    new_status = []
    for x in statuses:
        if not new_status:
            new_status.append(x)
        else:
            if new_status[-1]['status'] == x['status']:
                new_status[-1]['end'] = x['end']
            else:
                new_status.append(x)
    new_status[0]['begin'] = 0
    return new_status


def filter_statuses(statuses, duration_threshold, protect_initial=False):
    if duration_threshold != 0:  # Filter 0 intervals first
        statuses = filter_statuses(statuses, 0)
    if isinstance(duration_threshold, dict):
        statuses = [x for i, x in enumerate(statuses)
                    if x['status'] not in duration_threshold or
                    round(x['end'] - x['begin'], 1) >= duration_threshold[x['status']]
                    or i == len(statuses) - 1 or (protect_initial and i == 0)]
    else:
        statuses = [x for i, x in enumerate(statuses)
                    if round(x['end'] - x['begin'], 1) >= duration_threshold
                    or i == len(statuses) - 1 or (protect_initial and i == 0)]
    return coalesce_statuses(statuses)


class BaseAnnotator(object):
    batch_size = 100
    resize_factor = 1
    time_step = 0.1
    identifier = ''
    box_settings = ''

    def __init__(self, film_format, device, debug=False):
        self.device = device
        self.debug = debug
        self.params = BOX_PARAMETERS[film_format][self.box_settings]
        self.image_height = int(self.params['HEIGHT'] * self.resize_factor)
        self.image_width = int(self.params['WIDTH'] * self.resize_factor)
        self.shape = (self.batch_size, 3, int(self.params['HEIGHT'] * self.resize_factor),
                      int(self.params['WIDTH'] * self.resize_factor))
        self.to_predict = np.zeros(self.shape, dtype=np.uint8)
        self.process_index = 0
        self.begin_time = 0

    def process_frame(self, frame, time_point):
        #if time_point % self.time_step != 0:
        #    return
        box = frame[self.params['Y']: self.params['Y'] + self.params['HEIGHT'],
              self.params['X']: self.params['X'] + self.params['WIDTH']]
        if self.debug:
            cv2.imshow('frame_' + self.identifier, box)

        if self.resize_factor:
            box = cv2.resize(box, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        box = np.transpose(box, axes=(2, 0, 1))
        self.to_predict[self.process_index, ...] = box[None]
        self.process_index += 1

        if self.process_index == self.batch_size:
            self.annotate()
            self.reset(self.begin_time + (self.batch_size * self.time_step))

    def reset(self, begin_time):
        self.to_predict = np.zeros(self.shape, dtype=np.uint8)
        self.process_index = 0
        self.begin_time = begin_time

    def annotate(self):
        pass
