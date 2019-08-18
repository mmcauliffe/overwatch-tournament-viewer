import numpy as np
import cv2
from annotator.config import BOX_PARAMETERS


def filter_statuses(statuses, duration_threshold):
    statuses = [x for x in statuses if x['end'] - x['begin'] > duration_threshold]
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


class BaseAnnotator(object):
    batch_size = 100
    resize_factor = 1
    time_step = 0.1
    identifier = ''
    box_settings = ''

    def __init__(self, film_format, device):
        self.device = device
        self.params = BOX_PARAMETERS[film_format][self.box_settings]
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
        #cv2.imshow('frame', box)
        #cv2.waitKey()
        if self.resize_factor:
            box = cv2.resize(box, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        box = np.transpose(box, axes=(2, 0, 1))
        self.to_predict[self.process_index, ...] = box[None]
        self.process_index += 1

        if self.process_index == self.batch_size:
            self.annotate()
            self.to_predict = np.zeros(self.shape, dtype=np.uint8)
            self.process_index = 0
            self.begin_time += self.batch_size * self.time_step

    def annotate(self):
        pass
