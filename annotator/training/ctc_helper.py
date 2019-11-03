import torch
from torch.autograd import Variable
import os
import cv2
import numpy as np
from annotator.training.helper import Averager


def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def train_batch(net, train_iter, device, criterion, optimizer,image, spectator_modes, text, length):
    data = train_iter.next()

    d = data
    inputs = d[0]
    outputs = d[1]
    use_weights = False
    if len(d) > 2:
        use_weights = True
        weights = d[2][0].to(device)
    cpu_images = inputs['image'][0]
    cpu_specs = inputs['spectator_mode'][0]
    cpu_texts = outputs['the_labels'][0]
    cpu_lengths = outputs['label_length'][0]
    batch_size = cpu_images.size(0)
    if not batch_size:
        return
    loadData(image, cpu_images)
    loadData(spectator_modes, cpu_specs)
    loadData(text, cpu_texts)
    loadData(length, cpu_lengths)


    optimizer.zero_grad()
    preds = net(image, spectator_modes)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))

    cost = criterion(preds, text, preds_size, length)
    if use_weights:
        cost = (cost * weights / weights.sum()).sum()
    # net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def val(net, val_loader, device, criterion, working_dir, best_val_loss, converter, image, spectator_modes, text, length):
    print('Start val')
    n_test_disp = 10
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
            inputs = d[0]
            outputs = d[1]
            cpu_images = inputs['image'][0]
            cpu_specs = inputs['spectator_mode'][0]
            cpu_texts = outputs['the_labels'][0]
            cpu_lengths = outputs['label_length'][0]
            cpu_rounds = inputs['round'][0]

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
                if pred == target:
                    n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / (len(val_loader) * batch_size)
    print('Test loss: %f, accuracy: %f' % (loss_avg.val(), accuracy))

    if loss_avg.val() < best_val_loss:
        print('Saving new best model!')
        torch.save(net.state_dict(), os.path.join(working_dir, 'model.pth'))
        best_val_loss = loss_avg.val()

    assessment_file = os.path.join(working_dir, 'accuracy.txt')

    import datetime

    with open(assessment_file, 'a', encoding='utf8') as f:
        f.write('\n\n' + str(datetime.datetime.now()) + '\n')
        f.write('Test loss: %f, accuracy: %f\n' % (loss_avg.val(), accuracy))
    return best_val_loss
