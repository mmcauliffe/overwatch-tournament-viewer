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


import torch.nn as nn
from annotator.models import crnn
from annotator.training.helper import Averager, load_set

TEST = True
train_dir = r'N:\Data\Overwatch\training_data\player_ocr'

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

model_dir = r'N:\Data\Overwatch\models\player_ocr_test'
working_dir = r'N:\Data\Overwatch\models\debug\player_ocr'
shutil.rmtree(working_dir, ignore_errors=True)
os.makedirs(working_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'model.pth')


def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def test_errors():
    print('Start test')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()

    val_iter = iter(val_loader)
    with torch.no_grad():
        for index in range(len(val_loader)):
            print(index, len(val_loader))
            data = val_iter.next()

            d = data
            inputs, outputs = d[0], d[1]
            cpu_images = inputs['image'][0]
            cpu_texts = outputs['the_labels'][0]
            cpu_lengths = outputs['label_length'][0]
            cpu_rounds = inputs['round'][0]

            batch_size = cpu_images.size(0)
            if not batch_size:
                continue
            loadData(image, cpu_images)
            loadData(text, cpu_texts)
            loadData(length, cpu_lengths)


            preds = net(image)
            preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds = preds.to('cpu')
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            cpu_texts_decode = converter.decode(cpu_texts, cpu_lengths, raw=True)
            for i, (pred, target) in enumerate(zip(sim_preds, cpu_texts_decode)):
                if pred != target:
                    im = ((np.transpose(cpu_images[i, ...].numpy(), (1, 2, 0)) * 0.5) + 0.5) * 255
                    filename = '{} - {} - {}.jpeg'.format(cpu_rounds.numpy()[i],  target.replace(',', ''), pred.replace(',', ''))
                    filename = filename.replace(':', '').replace('ú', 'u').replace('ö', 'o').replace('!', '')
                    is_success, im_buf_arr = cv2.imencode(".jpg", im)
                    im_buf_arr.tofile(os.path.join(working_dir, filename))

if __name__ == '__main__':
    label_path = os.path.join(train_dir, 'labels_set.txt')
    label_set = load_set(label_path)
    for i, lab in enumerate(label_set):
        if not lab:
            blank_ind = i
            break
    else:
        blank_ind = len(label_set)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set = CTCHDF5Dataset(train_dir, batch_size, blank_ind, pre='train')
    test_set = CTCHDF5Dataset(train_dir, batch_size, blank_ind, pre='val')

    net = crnn.PlayerNameCRNN(label_set)
    converter = LabelConverter(label_set)

    if os.path.exists(model_path): # Initialize from CNN model
        d = torch.load(model_path)
        net.load_state_dict(d, strict=False)
        print('Loaded previous model')
    else:
        raise Exception('Could not find the model')


    image = torch.FloatTensor(batch_size, 3, image_height, image_width)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)
    if cuda and torch.cuda.is_available():
        net.cuda()
        image = image.cuda()
        #text = text.cuda()
    image = Variable(image)
    text = Variable(text)
    length = Variable(length)
    # loss averager
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                              shuffle=False, num_workers=0)
    print(len(val_loader), 'batches')

    test_errors()
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=False, num_workers=0)
    print(len(val_loader), 'batches')

    test_errors()
