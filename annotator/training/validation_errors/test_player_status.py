import os
import shutil
import torch
import numpy as np
import cv2

import time
import torch.optim as optim
import random
from annotator.datasets.cnn_dataset import CNNHDF5Dataset
import torch.nn as nn
from annotator.models.cnn import StatusCNN
from annotator.training.cnn_helper import train_batch, val
from annotator.training.helper import Averager, load_set

TEST = True
train_dir = r'E:\Data\Overwatch\training_data\player_status'

cuda = True
seed = 1
batch_size = 1200
image_height = 64
image_width = 64
num_channels = 3

manualSeed = 1234 # reproduce experiemnt

random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

model_dir = r'E:\Data\Overwatch\models\player_status'
working_dir = r'E:\Data\Overwatch\models\debug\player_status'
shutil.rmtree(working_dir, ignore_errors=True)
os.makedirs(working_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'model.pth')

def test_errors():
    print('Start test')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()

    val_iter = iter(val_loader)
    i = 0
    n_correct = 0
    loss_avg = Averager()
    corrects = {k: 0 for k in list(net.sets.keys())}

    total = 0
    label_corrects = {}
    label_totals = {}
    for k in corrects.keys():
        label_corrects[k] = list(0. for i in range(len(net.sets[k])))
        label_totals[k] = list(0. for i in range(len(net.sets[k])))
    with torch.no_grad():
        for index in range(len(val_loader)):
            print(index, len(val_loader))
            data = val_iter.next()

            inputs, labels = data
            for k, v in inputs.items():
                if k in ['round', 'time_point']:
                    inputs[k] = v[0]
                    continue
                inputs[k] = v[0].float().to(device)
            for k, v in labels.items():
                labels[k] = v[0].long().to(device)
            predicteds = net(inputs)

            loss = None

            for k, v in predicteds.items():
                if loss is None:
                    loss = losses[k](v, labels[k])
                else:
                    loss += losses[k](v, labels[k])
            loss_avg.add(loss)
            errors = {}
            for k, v in predicteds.items():
                if k not in errors:
                    errors[k] = []
                _, predicteds[k] = torch.max(v, 1)
                corrects[k] += (predicteds[k] == labels[k]).sum().item()
                c = (predicteds[k] == labels[k]).squeeze().to('cpu')

                if c.shape:
                    for i in range(c.shape[0]):
                        label = labels[k][i]
                        label_corrects[k][label] += c[i].item()
                        label_totals[k][label] += 1
                        if c[i].item() == 0:
                            errors[k].append({'image': inputs['image'][i, ...].cpu().numpy(),
                                           'truth': net.sets[k][label],
                                           'predicted': net.sets[k][predicteds[k][i]],
                                           'round': inputs['round'][i].item(),
                                           'time_point': inputs['time_point'][i].item()})
                else:
                    label = labels[k][0]
                    label_corrects[k][label] += c.item()
                    label_totals[k][label] += 1
            total += inputs['image'].size(0)
            for k, e in errors.items():
                for error in e:
                    im = ((np.transpose(error['image'], (1, 2, 0)) * 0.5) + 0.5) * 255
                    directory = os.path.join(working_dir, k, error['truth'].replace(':', '').replace(':', '').replace('ú', 'u').replace('ö', 'o').replace('/', ''))
                    if not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
                    minutes = int(error['time_point'] / 60)
                    seconds = round(error['time_point'] - int(minutes * 60), 1)
                    filename = '{} - {}_{} - predicted_{}.jpeg'.format(error['round'],
                                                                    minutes, seconds,
                                                                    error['predicted']).replace(':', '').replace('ú', 'u').replace('ö', 'o').replace('/', '')
                    cv2.imwrite(os.path.join(directory, filename), im)

    print('Val loss: {}'.format(loss_avg.val()))
    for k, v in corrects.items():
        print('%s accuracy of the network on the %d test images: %d %%' % (k, total,
            100 * corrects[k] / total))

    assessment_file = os.path.join(working_dir, 'accuracy.txt')

    import datetime

    with open(assessment_file, 'a', encoding='utf8') as f:
        f.write('\n\n' + str(datetime.datetime.now()) + '\n')
        for k in corrects.keys():
            print(k)
            for i in range(len(net.sets[k])):
                if not label_totals[k][i]:
                    continue
                print('Accuracy of %5s : %2d %% (%d / %d)' % (
                    net.sets[k][i], 100 * label_corrects[k][i] / label_totals[k][i], label_corrects[k][i], label_totals[k][i]))
                print('Accuracy of %5s : %2d %% (%d / %d)' % (
                    net.sets[k][i], 100 * label_corrects[k][i] / label_totals[k][i], label_corrects[k][i], label_totals[k][i]), file=f)

if __name__ == '__main__':
    input_set_files = {
        'color': os.path.join(train_dir, 'color_set.txt'),
        'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),
         }
    input_sets = {}
    for k, v in input_set_files.items():
        input_sets[k] = load_set(v)

    set_files = {  # 'player': os.path.join(train_dir, 'player_set.txt'),
        'hero': os.path.join(train_dir, 'hero_set.txt'),
        'alive': os.path.join(train_dir, 'alive_set.txt'),
        'ult': os.path.join(train_dir, 'ult_set.txt'),
        #'switch': os.path.join(train_dir, 'switch_set.txt'),
        'status': os.path.join(train_dir, 'status_set.txt'),
        'antiheal': os.path.join(train_dir, 'antiheal_set.txt'),
        'immortal': os.path.join(train_dir, 'immortal_set.txt'),
    }
    sets = {}
    for k, v in set_files.items():
        sets[k] = load_set(v)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    test_set = CNNHDF5Dataset(train_dir, sets=sets, input_sets=input_sets, batch_size=batch_size, pre='val')

    net = StatusCNN(sets, input_sets)
    net.to(device)

    if os.path.exists(model_path): # Initialize from CNN model
        d = torch.load(model_path)
        net.load_state_dict(d, strict=False)
        print('Loaded previous model')

    losses = {}
    for k in sets.keys():
        losses[k] = nn.CrossEntropyLoss()
        losses[k].to(device)


    # loss averager
    loss_avg = Averager()
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=True, num_workers=0)
    print(len(val_loader), 'batches')

    test_errors()
