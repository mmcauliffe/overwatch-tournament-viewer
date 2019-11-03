import os
import shutil
import torch
import numpy as np
import cv2

import time
from torch.nn import CTCLoss
from torch.autograd import Variable
import random
from annotator.datasets.ctc_dataset import CTCHDF5Dataset, LabelConverter
from annotator.game_values import HERO_SET, COLOR_SET, ABILITY_SET, KILL_FEED_INFO

ability_mapping = KILL_FEED_INFO['ability_mapping']
npc_set = KILL_FEED_INFO['npc_set']
deniable_ults = KILL_FEED_INFO['deniable_ults']
denying_abilities = KILL_FEED_INFO['denying_abilities']
npc_mapping = KILL_FEED_INFO['npc_mapping']

import torch.nn as nn
from annotator.models import crnn
from annotator.training.helper import Averager, load_set

TEST = True
train_dir = r'E:\Data\Overwatch\training_data\kill_feed_ctc'

cuda = True
seed = 1
batch_size = 100
image_height = 64
image_width = 64
num_channels = 3

manualSeed = 1234 # reproduce experiemnt

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

model_dir = r'E:\Data\Overwatch\models\kill_feed_ctc'
working_dir = r'E:\Data\Overwatch\models\debug\kill_feed'
shutil.rmtree(working_dir, ignore_errors=True)
os.makedirs(working_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'model.pth')


def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def convert_kf_ctc_output(ret, show=False):
    if show:
        print(ret)
    data = {'first_hero': 'n/a',
            'first_color': 'n/a',
            'assists': set([]),
            'ability': 'n/a',
            'headshot': 'n/a',
            'second_hero': 'n/a',
            'second_color': 'n/a'}
    first_intervals = []
    second_intervals = []
    ability_intervals = []
    for i in ret:
        if i in ABILITY_SET:
            ability_intervals.append(i)
        if not len(ability_intervals):
            first_intervals.append(i)
        elif i not in ABILITY_SET:
            second_intervals.append(i)
    for i in first_intervals:
        if i in COLOR_SET + ['nonwhite']:
            data['first_color'] = i
        elif i in HERO_SET:
            data['first_hero'] = i
        else:
            if i not in data['assists'] and i.replace('_assist', '') != data['first_hero'] and not i.endswith('npc'):
                data['assists'].add(i)
    for i in ability_intervals:
        if i.endswith('headshot'):
            data['headshot'] = True
            data['ability'] = i.replace(' headshot', '')
        else:
            data['ability'] = i
            data['headshot'] = False
    for i in second_intervals:
        i = i.replace('_assist', '')
        if i in COLOR_SET + ['nonwhite']:
            data['second_color'] = i
        elif i.endswith('_npc'):
            data['second_hero'] = i.replace('_npc', '')
        elif i in HERO_SET:
            data['second_hero'] = i
    if data['first_hero'] != 'n/a':
        if data['ability'] not in ability_mapping[data['first_hero']]:
            data['ability'] = 'primary'
    return data

def test_errors():
    print('Start test')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()

    val_iter = iter(val_loader)
    i = 0
    n_correct = 0
    loss_avg = Averager()
    with torch.no_grad():
        for index in range(len(val_loader)):
            data = val_iter.next()

            d = data
            inputs, outputs = d[0], d[1]
            cpu_images = inputs['image'][0]
            cpu_specs = inputs['spectator_mode'][0]
            cpu_texts = outputs['the_labels'][0]
            cpu_lengths = outputs['label_length'][0]
            cpu_rounds = inputs['round'][0]
            cpu_time_points = inputs['time_point'][0]

            batch_size = cpu_images.size(0)
            if not batch_size:
                continue
            loadData(image, cpu_images)
            loadData(spectator_modes, cpu_specs)
            loadData(text, cpu_texts)
            loadData(length, cpu_lengths)


            preds = net(image, spectator_modes)
            preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))

            cost = criterion(preds, text, preds_size, length) / batch_size
            loss_avg.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds = preds.to('cpu')
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            cpu_texts_decode = converter.decode(cpu_texts, cpu_lengths, raw=True)
            for i, (pred, target) in enumerate(zip(sim_preds, cpu_texts_decode)):
                if convert_kf_ctc_output(pred) != convert_kf_ctc_output(target):
                    im = np.transpose(cpu_images[i, ...].numpy(), (1, 2, 0))
                    spec_mode = spectator_mode_set[cpu_specs[i]]
                    directory = os.path.join(working_dir, spec_mode)
                    if not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
                    filename = '{} - {} - {} - {}.jpeg'.format(cpu_rounds.numpy()[i],cpu_time_points.numpy()[i], target.replace(',', ' '), pred.replace(',', ' '))
                    filename = filename.replace(':', '').replace('ú', 'u').replace('ö', 'o').replace('!', '')
                    cv2.imwrite(os.path.join(directory, filename), im)

if __name__ == '__main__':
    label_path = os.path.join(train_dir, 'labels_set.txt')
    spec_mode_path = os.path.join(train_dir, 'spectator_mode_set.txt')
    spectator_mode_set = load_set(spec_mode_path)
    label_set = load_set(label_path)
    for i, lab in enumerate(label_set):
        if not lab:
            blank_ind = i
            break
    else:
        blank_ind = len(label_set)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    test_set = CTCHDF5Dataset(train_dir, batch_size, blank_ind, pre='val')

    net = crnn.KillFeedCRNN(label_set, spectator_mode_set)
    converter = LabelConverter(label_set)
    criterion = CTCLoss()

    if os.path.exists(model_path): # Initialize from CNN model
        d = torch.load(model_path)
        net.load_state_dict(d, strict=False)
        print('Loaded previous model')


    image = torch.FloatTensor(batch_size, 3, image_height, image_width)
    spectator_modes = torch.IntTensor(batch_size)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)
    if cuda and torch.cuda.is_available():
        net.cuda()
        image = image.cuda()
        spectator_modes = spectator_modes.cuda()
        criterion = criterion.cuda()
        #text = text.cuda()
    image = Variable(image)
    spectator_modes = Variable(spectator_modes)
    text = Variable(text)
    length = Variable(length)
    # loss averager
    loss_avg = Averager()
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=True, num_workers=0)
    print(len(val_loader), 'batches')

    test_errors()
