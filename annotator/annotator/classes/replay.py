import os
import torch
from annotator.annotator.classes.pause import PauseAnnotator
from annotator.models.cnn import ReplayCNN
from annotator.training.helper import load_set


class ReplayAnnotator(PauseAnnotator):
    time_step = 1
    resize_factor = 0.5
    identifier = 'replay'
    box_settings = 'REPLAY'
    cnn = ReplayCNN

    def generate_replays(self):
        status = self.generate_final_statuses()
        print('replay', status)
        events = []
        for interval in status:
            if interval['status'] == self.identifier:
                events.append(interval)
        return events