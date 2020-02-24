import requests
import os
import numpy as np
import random
import cv2
import shutil

import sys
import traceback
from multiprocessing import Process, Manager, Queue, cpu_count, Value, Lock, JoinableQueue
import time
from queue import Empty, Full

from annotator.config import offsets
from annotator.utils import get_duration, FileVideoStream, get_vod_path

from annotator.api_requests import get_train_vods


training_data_directory = r'N:\Data\Overwatch\training_data'

cnn_status_train_dir = os.path.join(training_data_directory, 'player_status_cnn')
ocr_status_train_dir = os.path.join(training_data_directory, 'player_status_ocr')
lstm_status_train_dir = os.path.join(training_data_directory, 'player_status_lstm')
cnn_mid_train_dir = os.path.join(training_data_directory, 'mid_cnn')
lstm_mid_train_dir = os.path.join(training_data_directory, 'mid_lstm')
cnn_kf_train_dir = os.path.join(training_data_directory, 'kf_cnn')
lstm_kf_train_dir = os.path.join(training_data_directory, 'kf_lstm')


class Counter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


class Stopped(object):
    def __init__(self, initval=False):
        self.val = Value('i', initval)
        self.lock = Lock()

    def stop(self):
        with self.lock:
            self.val.value = True

    def stop_check(self):
        with self.lock:
            return self.val.value


class AnalysisWorker(Process):
    resize_factor = 0.2
    def __init__(self, job_q, sets, states, return_dict, counter, stopped, ignore_errors=False):
        Process.__init__(self)
        self.job_q = job_q
        self.sets = sets
        self.states = states
        self.return_dict = return_dict
        self.counter = counter
        self.stopped = stopped
        self.ignore_errors = ignore_errors

    def lookup_data(self, time_point):
        d = {}
        beginning = False

        for k, s in self.sets.items():
            for interval in self.states[k]:
                if interval['begin'] <= time_point < interval['end']:
                    d[k] = interval['status']
                    if time_point == interval['begin']:
                        beginning = True
                    break
            else:
                if time_point == self.states[k][-1]['end']:
                    beginning = True
                d[k] = s[0]
        if d['game'] == 'pause':
            d['left'] = 'n/a'
            d['right'] = 'n/a'
            d['map'] = 'n/a'
            d['submap'] = 'n/a'
            d['left_color'] = 'n/a'
            d['right_color'] = 'n/a'
            d['attacking_side'] = 'n/a'
        elif d['game'] == 'replay':
            d['left'] = 'not_zoom'
            d['right'] = 'not_zoom'
        elif d['game'] == 'smaller_window':
            d['left'] = 'n/a'
            d['right'] = 'n/a'
        return d, beginning

    def run(self):
        stream = None
        actual_duration, mode = None, None
        skip_duration_lookup = False
        while True:
            self.counter.increment()
            try:
                arguments = self.job_q.get(timeout=1)
            except Empty:
                break
            self.job_q.task_done()
            if self.stopped.stop_check():
                continue
            try:
                v, time_point = arguments
                if stream is None:
                    stream = cv2.VideoCapture(get_vod_path(v))
                    stream.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                    duration = stream.get(cv2.CAP_PROP_POS_MSEC) / 1000
                d, ignore = self.lookup_data(time_point)
                if not ignore:
                    offset = 0
                    if v['id'] in offsets:
                        offset = offsets[v['id']]
                    if not skip_duration_lookup and actual_duration is None and v['channel']['site'] == 'Y':
                        actual_duration, mode = get_duration(v)
                        if actual_duration is None:
                            skip_duration_lookup = True
                    if offset:
                        actual_time_point = time_point - offset
                    elif actual_duration:
                        if abs(actual_duration - duration) > 2:
                            actual_time_point = time_point / actual_duration
                            actual_time_point *= int(duration)
                            #print(time_point, actual_time_point)
                            #print(self.duration,  self.actual_duration)
                        else:
                            actual_time_point = time_point
                    else:
                        actual_time_point = time_point
                    stream.set(cv2.CAP_PROP_POS_MSEC, int(actual_time_point*1000))
                    (grabbed, frame) = stream.read()
                    if not grabbed:
                        ignore = True
                    else:
                        #cv2.imshow('frame_{}'.format(time_point), frame)
                        #cv2.waitKey()
                        box = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
                        # cv2.imshow('frame', box)
                        # print(box.shape)
                        # cv2.waitKey()
                        box = np.transpose(box, axes=(2, 0, 1))
                        d['img'] = box
                self.return_dict[(v['id'], time_point)] = (d, ignore)
            except Exception as e:
                if self.ignore_errors:
                    continue
                self.stopped.stop()
                self.return_dict['error'] = arguments, Exception(traceback.format_exception(*sys.exc_info()))
        if stream is not None:
            stream.release()
        return



def generate_cache(v, num_procs, call_back, stop_check):
    g = GameGenerator()
    g.add_new_round_info(v)
    if not g.generate_data:
        return
    stopped = Stopped()
    job_queue = JoinableQueue(100)
    time_point = 0
    while time_point < g.expected_duration:
        try:
            job_queue.put((v, time_point), False)
        except Full:
            break
        ts = g.time_step
        if g.special_time_steps:
            found = False
            for k, intervals in g.special_time_steps.items():
                for interval in intervals:
                    if interval['begin'] <= time_point < interval['end']:
                        ts = k
                        if time_point + ts > interval['end']:
                            time_point = interval['end'] - ts
                        found = True
                        break
                if found:
                    break
        time_point += ts

    manager = Manager()
    return_dict = manager.dict()
    procs = []

    counter = Counter()
    for i in range(num_procs):
        p = AnalysisWorker(job_queue, g.sets, g.states, return_dict, counter, stopped)
        procs.append(p)
        p.start()
    if call_back is not None:
        call_back('Performing analysis...')

    while time_point < g.expected_duration:
        if stop_check is not None and stop_check():
            stopped.stop()
            time.sleep(1)
            break
        job_queue.put((v, time_point))

        if call_back is not None:
            value = counter.value()
            call_back(value)
        ts = g.time_step
        if g.special_time_steps:
            found = False
            for k, intervals in g.special_time_steps.items():
                for interval in intervals:
                    if interval['begin'] <= time_point < interval['end']:
                        ts = k
                        if time_point + ts > interval['end']:
                            time_point = interval['end'] - ts
                        found = True
                        break
                if found:
                    break
        time_point += ts
    job_queue.join()

    for p in procs:
        p.join()
    if 'error' in return_dict:
        element, exc = return_dict['error']
        print(element)
        raise exc
    to_return = {}
    to_return.update(return_dict)
    g.update_from_dict(return_dict)
    g.cleanup_round()

def generate_data_for_game_cnn_mp(vods):
    for round_index, v in enumerate(vods):
        begin_time = time.time()
        print("Processing vod {} of {}".format(round_index, len(vods)))
        print(v)
        num_procs = 3
        generate_cache(v, num_procs, None, None)
        print('Finished in {} seconds!'.format(time.time() - begin_time))


def generate_data_for_game_cnn(vods):

    import time as timepackage
    generators = [GameGenerator()]
    checked_vods = set()
    average_times = [0, 0, 0, 0]
    for round_index, v in enumerate(vods):
        if v['id'] in checked_vods:
            continue
        checked_vods.add(v['id'])
        print("Processing vod {} of {}".format(round_index, len(vods)))
        print(v)
        begin_time = timepackage.time()
        process_vod = False
        for i, g in enumerate(generators):
            g.add_new_round_info(v)
            if g.generate_data:
                process_vod = True
        if not process_vod:
            print('skipping!')
            continue
        time_step = min(x.minimum_time_step for x in generators if x.generate_data)
        actual_duration = None
        mode = None
        #if v['channel']['site'] == 'Y':
        #    actual_duration, mode = get_duration(v)
        offset = 0
        if v['id'] in offsets:
            offset = offsets[v['id']]
        fvs = FileVideoStream(get_vod_path(v), offset, 0, generators[0].broadcast_event_time_step,
                              real_begin=0, special_time_steps=generators[0].special_time_steps,
                              actual_duration=actual_duration, offset=offset, mode=mode).start()
        timepackage.sleep(1.0)
        frame_ind = 0
        end = fvs.end
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            frame = frame['frame']
            for i, g in enumerate(generators):
                begin = timepackage.time()
                g.process_frame(frame, time_point, frame_ind)
                average_times[i] += (timepackage.time()-begin)/100

            if frame_ind % 100 == 0:
                print('Frame: {}/{}'.format(time_point, end))
                for i, g in enumerate(generators):
                    print('Average process frame time for {}:'.format(type(g).__name__), average_times[i])
                average_times = [0, 0, 0, 0]
            frame_ind += 1

        for g in generators:
            g.cleanup_round()
        print('Finished in {} seconds!'.format(timepackage.time() - begin_time))


if __name__ == '__main__':


    from annotator.data_generation.classes import GameGenerator
    #rounds_plus = get_train_rounds_plus()

    vods = get_train_vods()#[:max_count]
    max_count = 2

    #FILTER
    #rounds = get_train_rounds()
    #rounds = [r for r in rounds if r['stream_vod'] is not None and r['stream_vod']['film_format'] == 'A']
    #rounds = rounds[:max_count]
    #hero_times = get_hero_play_time(rounds)
    #analyze_missing_vods(rounds, vods)

    #generate_data_for_game_cnn(vods)
    generate_data_for_game_cnn_mp(vods)


