import torch
from torch.autograd import Variable
import os
import shutil
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import CTCLoss
from annotator.training.helper import Averager, load_set, load_checkpoint, lr, beta1, display_interval, print_params, \
    set_train, load_checkpoint, weights_init
from annotator.game_values import HERO_SET, COLOR_SET, ABILITY_SET, KILL_FEED_INFO


ability_mapping = KILL_FEED_INFO['ability_mapping']
npc_set = KILL_FEED_INFO['npc_set']
deniable_ults = KILL_FEED_INFO['deniable_ults']
denying_abilities = KILL_FEED_INFO['denying_abilities']
npc_mapping = KILL_FEED_INFO['npc_mapping']

def loadData(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def train_model(working_dir, train_dir, model_class, data_class, label_set, spec_modes=None,
                    early_stopping_threshold=2, batch_size=100,
                test_batch_size=200, num_epochs=50, image_height=32, image_width=296, kill_feed=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, lab in enumerate(label_set):
        if not lab:
            blank_ind = i
            break
    else:
        blank_ind = len(label_set)

    train_set = data_class(train_dir, batch_size, blank_ind,
                                       pre='train', modes=spec_modes)
    test_set = data_class(train_dir, test_batch_size, blank_ind,
                                      pre='val', modes=spec_modes)
    criterion = CTCLoss()
    image = torch.FloatTensor(batch_size, 3, image_height, image_width)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)
    criterion = criterion.cuda()
    image = image.cuda()
    image = Variable(image)
    text = Variable(text)
    length = Variable(length)
    if kill_feed:
        left_colors = torch.FloatTensor(batch_size, 3)
        right_colors = torch.FloatTensor(batch_size, 3)
        left_colors = left_colors.cuda()
        right_colors = right_colors.cuda()
        left_colors = Variable(left_colors)
        right_colors = Variable(right_colors)

    net = model_class(label_set)

    net = set_train(net, True)
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters, lr=lr, betas=(beta1, 0.999))

    print_params(net)

    # loss averager
    loss_avg = Averager()
    import time
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=4,
                                               shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                             shuffle=True, num_workers=4, pin_memory=True)
    print(len(train_loader), 'batches')
    check_point_path = os.path.join(working_dir, 'checkpoint.pth')
    net, optimizer, start_epoch, best_val_loss = load_checkpoint(net, optimizer, check_point_path)

    net = net.to(device)
    # now individually transfer the optimizer parts...
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    best_val_loss = np.inf
    last_improvement = start_epoch
    loss_path = os.path.join(working_dir, 'losses.txt')
    if start_epoch == 0:
        net.apply(weights_init)
    for epoch in range(start_epoch, num_epochs):
        print('Epoch', epoch)
        begin = time.time()
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            net = set_train(net, True)
            if kill_feed:
                cost = train_batch_kf(net, train_iter, criterion, optimizer, image, left_colors,
                                      right_colors, text, length)
            else:
                cost = train_batch_ocr(net, train_iter, criterion, optimizer, image, text, length)
            loss_avg.add(cost)
            i += 1

            if i % display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        prev_best = best_val_loss
        with open(loss_path, 'a') as f:
            f.write('Epoch {}: '.format(epoch))
        if kill_feed:
            best_val_loss = val_kf(epoch, net, val_loader, criterion, working_dir, best_val_loss, image,
                                   left_colors, right_colors, text, length)
        else:
            best_val_loss = val_ocr(net, val_loader, criterion, working_dir, best_val_loss,image, text, length)
        if best_val_loss < prev_best:
            last_improvement = epoch

        # do checkpointing
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'best_val_loss': best_val_loss}
        torch.save(state, check_point_path)
        time_taken = time.time() - begin
        print('Time per epoch: {} seconds'.format(time_taken))
        with open(loss_path, 'a') as f:
            f.write('Time per epoch: {}\n\n'.format(time_taken))
        if epoch - last_improvement == early_stopping_threshold:
            print('Stopping training, val loss hasn\'t improved in {} iterations.'.format(early_stopping_threshold))
            break
    print('Completed training, best val loss was: {}'.format(best_val_loss))

def convert_kf_ctc_output(ret, show=False):
    if show:
        print(ret)
    data = {'first_hero': 'n/a',
            'first_side': 'n/a',
            'assists': set([]),
            'ability': 'n/a',
            'headshot': 'n/a',
            'second_hero': 'n/a',
            'second_side': 'n/a'}
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
        if i in ['right', 'left']:
            data['first_side'] = i
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
        if i in ['right', 'left']:
            data['second_side'] = i
        elif i.endswith('_npc'):
            data['second_hero'] = i.replace('_npc', '')
        elif i in HERO_SET:
            data['second_hero'] = i
    if data['first_hero'] != 'n/a':
        if data['ability'] not in ability_mapping[data['first_hero']]:
            data['ability'] = 'primary'
    return data

def test_errors(model_dir, working_dir, train_dir, model_class, data_class, label_set, spec_mode=None,
                batch_size=600, image_height=32, image_width=296, kill_feed=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Processing ' + spec_mode)
    for i, lab in enumerate(label_set):
        if not lab:
            blank_ind = i
            break
    else:
        blank_ind = len(label_set)

    image = torch.FloatTensor(batch_size, 3, image_height, image_width)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)
    image = image.cuda()
    image = Variable(image)
    text = Variable(text)
    length = Variable(length)
    if kill_feed:
        left_colors = torch.FloatTensor(batch_size, 3)
        right_colors = torch.FloatTensor(batch_size, 3)
        left_colors = left_colors.cuda()
        right_colors = right_colors.cuda()
        left_colors = Variable(left_colors)
        right_colors = Variable(right_colors)
    net = model_class(label_set)
    net.to(device)
    base_model_path = os.path.join(model_dir, 'model.pth')
    if spec_mode:
        spec_mode_model_path = os.path.join(model_dir, spec_mode, 'model.pth')
        if os.path.exists(spec_mode_model_path):  # Initialize from CNN model
            d = torch.load(spec_mode_model_path)
            net.load_state_dict(d, strict=False)
            print('Loaded {} model'.format(spec_mode))
        elif os.path.exists(base_model_path):  # Initialize from CNN model
            d = torch.load(base_model_path)
            net.load_state_dict(d, strict=False)
            print('Loaded base model')
        else:
            raise Exception('Could not find the model')
    elif os.path.exists(base_model_path):  # Initialize from CNN model
        d = torch.load(base_model_path)
        net.load_state_dict(d, strict=False)
        print('Loaded base model')
    else:
        raise Exception('Could not find the model')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    for pre in ['train', 'val']:
        if spec_mode:
            test_set = data_class(train_dir, batch_size, blank_ind, pre=pre,
                                              modes=[spec_mode])
        else:
            test_set = data_class(train_dir, batch_size, blank_ind, pre=pre)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                 shuffle=False, num_workers=0)
        print(len(val_loader), 'batches')

        val_iter = iter(val_loader)
        loss_avg = Averager()
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
                cpu_time_points = inputs['time_point'][0]

                batch_size = cpu_images.size(0)
                if not batch_size:
                    continue
                loadData(image, cpu_images)
                if kill_feed:
                    cpu_left_colors = inputs['left_color'][0]
                    cpu_right_colors = inputs['right_color'][0]
                    loadData(left_colors, cpu_left_colors)
                    loadData(right_colors, cpu_right_colors)
                loadData(text, cpu_texts)
                loadData(length, cpu_lengths)
                if kill_feed:
                    preds = net(image, left_colors, right_colors)
                preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))

                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds = preds.to('cpu')
                sim_preds = net.converter.decode(preds.data, preds_size.data, raw=False)
                cpu_texts_decode = net.converter.decode(cpu_texts, cpu_lengths, raw=True)
                for i, (pred, target) in enumerate(zip(sim_preds, cpu_texts_decode)):
                    if kill_feed:
                        accuracy = convert_kf_ctc_output(pred) == convert_kf_ctc_output(target)
                    if not accuracy:
                        im = ((np.transpose(cpu_images[i, ...].numpy(), (1, 2, 0)) * 0.5) + 0.5) * 255
                        directory = os.path.join(working_dir, spec_mode)
                        if not os.path.exists(directory):
                            os.makedirs(directory, exist_ok=True)
                        time_point = cpu_time_points.numpy()[i]
                        minutes = int(time_point / 60)
                        seconds = round(time_point - int(minutes * 60), 1)
                        filename = '{} - {}_{} - {} - {}.jpeg'.format(cpu_rounds.numpy()[i], minutes, seconds,
                                                                      target.replace(',', ' '), pred.replace(',', ' '))
                        filename = filename.replace(':', '').replace('ú', 'u').replace('ö', 'o').replace('!', '')
                        cv2.imwrite(os.path.join(directory, filename), im)

        print('Val loss: {}'.format(loss_avg.val()))



def fine_tune_model(working_dir, train_dir, model_class, data_class, label_set, spec_mode,
                    early_stopping_threshold=2, batch_size=100,
                    test_batch_size=200, num_epochs=50, image_height=32, image_width=296, kill_feed=True):

    base_model_path = os.path.join(working_dir, 'model.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ft_working_dir = os.path.join(working_dir, spec_mode)
    if os.path.exists(ft_working_dir):
        return
    os.makedirs(ft_working_dir, exist_ok=True)

    for i, lab in enumerate(label_set):
        if not lab:
            blank_ind = i
            break
    else:
        blank_ind = len(label_set)
    train_set = data_class(train_dir, batch_size, blank_ind,
                                       pre='train', modes=[spec_mode])
    test_set = data_class(train_dir, test_batch_size, blank_ind,
                                      pre='val', modes=[spec_mode])
    criterion = CTCLoss()
    image = torch.FloatTensor(batch_size, 3, image_height, image_width)
    text = torch.IntTensor(batch_size * 5)
    length = torch.IntTensor(batch_size)
    criterion = criterion.cuda()
    image = image.cuda()
    image = Variable(image)
    text = Variable(text)
    length = Variable(length)
    if kill_feed:
        left_colors = torch.FloatTensor(batch_size, 3)
        right_colors = torch.FloatTensor(batch_size, 3)
        left_colors = left_colors.cuda()
        right_colors = right_colors.cuda()
        left_colors = Variable(left_colors)
        right_colors = Variable(right_colors)

    net = model_class(label_set)

    if os.path.exists(base_model_path):  # Initialize from CNN model
        d = torch.load(base_model_path)
        net.load_state_dict(d, strict=False)
        print('Initialized from base model!')

    net = set_train(net, True, fine_tune=True)
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.Adam(parameters, lr=lr, betas=(beta1, 0.999))

    print_params(net)

    # loss averager
    loss_avg = Averager()
    import time
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=0,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                             shuffle=True, num_workers=0)
    print(len(train_loader), 'batches')
    check_point_path = os.path.join(ft_working_dir, 'checkpoint.pth')
    net, optimizer, start_epoch, best_val_loss = load_checkpoint(net, optimizer, check_point_path)

    net = net.to(device)
    # now individually transfer the optimizer parts...
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    best_val_loss = np.inf
    last_improvement = start_epoch
    if start_epoch == 0:
        if kill_feed:
            best_val_loss = val_kf(-1, net, val_loader, criterion, ft_working_dir, best_val_loss, image,
                                   left_colors, right_colors, text, length)
        else:
            best_val_loss = val_ocr(net, val_loader, criterion, ft_working_dir, best_val_loss,image, text, length)
        print('Base model val loss is {}'.format(best_val_loss))
    loss_path = os.path.join(ft_working_dir, 'losses.txt')
    with open(loss_path, 'a') as f:
        f.write('Base model val loss is {} over {} batches\n\n'.format(best_val_loss, len(train_loader)))
    for epoch in range(start_epoch, num_epochs):
        print('Epoch', epoch)
        begin = time.time()
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            net = set_train(net, True, fine_tune=True)
            if kill_feed:
                cost = train_batch_kf(net, train_iter, criterion, optimizer, image, left_colors,
                                      right_colors, text, length)
            else:
                cost = train_batch_ocr(net, train_iter, criterion, optimizer, image, text, length)
            loss_avg.add(cost)
            i += 1

            if i % display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        prev_best = best_val_loss
        with open(loss_path, 'a') as f:
            f.write('Epoch {}: '.format(epoch))
        if kill_feed:
            best_val_loss = val_kf(epoch, net, val_loader, criterion, ft_working_dir, best_val_loss, image,
                                   left_colors, right_colors, text, length)
        else:
            best_val_loss = val_ocr(net, val_loader, criterion, ft_working_dir, best_val_loss,image, text, length)
        if best_val_loss < prev_best:
            last_improvement = epoch

        # do checkpointing
        state = {'epoch': epoch + 1, 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'best_val_loss': best_val_loss}
        torch.save(state, check_point_path)
        time_taken = time.time() - begin
        print('Time per epoch: {} seconds'.format(time_taken))
        with open(loss_path, 'a') as f:
            f.write('Time per epoch: {}\n\n'.format(time_taken))
        if epoch - last_improvement == early_stopping_threshold:
            print('Stopping training, val loss hasn\'t improved in {} iterations.'.format(early_stopping_threshold))
            break
    print('Completed training, best val loss was: {}'.format(best_val_loss))


def train_batch_kf(net, train_iter, criterion, optimizer, image, left_colors, right_colors, text, length):
    data = train_iter.next()

    d = data
    inputs = d[0]
    outputs = d[1]
    cpu_images = inputs['image'][0]
    cpu_left_colors = inputs['left_color'][0]
    cpu_right_colors = inputs['right_color'][0]
    cpu_texts = outputs['the_labels'][0]
    cpu_lengths = outputs['label_length'][0]
    batch_size = cpu_images.size(0)
    if not batch_size:
        return
    loadData(image, cpu_images)
    loadData(left_colors, cpu_left_colors)
    loadData(right_colors, cpu_right_colors)
    loadData(text, cpu_texts)
    loadData(length, cpu_lengths)


    optimizer.zero_grad()
    preds = net(image, left_colors, right_colors)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))

    cost = criterion(preds, text, preds_size, length)

    # net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def train_batch_ocr(net, train_iter, criterion, optimizer, image, text, length):
    data = train_iter.next()

    d = data
    inputs = d[0]
    outputs = d[1]
    use_weights = False
    #if len(d) > 2:
    #    use_weights = True
    #    weights = d[2][0].to(device)
    cpu_images = inputs['image'][0]
    cpu_texts = outputs['the_labels'][0]
    cpu_lengths = outputs['label_length'][0]
    batch_size = cpu_images.size(0)
    if not batch_size:
        return
    loadData(image, cpu_images)
    loadData(text, cpu_texts)
    loadData(length, cpu_lengths)


    optimizer.zero_grad()
    preds = net(image)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length)
    #if use_weights:
    #    cost = (cost * weights / weights.sum()).sum()
    # net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def val_kf(epoch, net, val_loader, criterion, working_dir, best_val_loss, image, left_colors, right_colors, text, length):
    print('Start val')
    n_test_disp = 10
    for p in net.parameters():
        p.requires_grad = False

    net.eval()

    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    n_total = 0
    loss_avg = Averager()
    with torch.no_grad():
        for index in range(len(val_loader)):
            data = val_iter.next()

            d = data
            inputs = d[0]
            outputs = d[1]
            cpu_images = inputs['image'][0]
            cpu_left_colors = inputs['left_color'][0]
            cpu_right_colors = inputs['right_color'][0]
            cpu_texts = outputs['the_labels'][0]
            cpu_lengths = outputs['label_length'][0]
            cpu_rounds = inputs['round'][0]

            batch_size = cpu_images.size(0)
            if not batch_size:
                continue
            loadData(image, cpu_images)
            loadData(left_colors, cpu_left_colors)
            loadData(right_colors, cpu_right_colors)
            loadData(text, cpu_texts)
            loadData(length, cpu_lengths)

            preds = net(image, left_colors, right_colors)
            preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))

            cost = criterion(preds, text, preds_size, length)
            loss_avg.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds = preds.to('cpu')
            sim_preds = net.converter.decode(preds.data, preds_size.data, raw=False)
            cpu_texts_decode = net.converter.decode(cpu_texts, cpu_lengths, raw=True)
            n_total += batch_size
            for i, (pred, target) in enumerate(zip(sim_preds, cpu_texts_decode)):
                if pred == target:
                    n_correct += 1

    raw_preds = net.converter.decode(preds.data, preds_size.data, raw=True)[:n_test_disp]
    example_dir = os.path.join(working_dir, 'examples', str(epoch))
    os.makedirs(example_dir, exist_ok=True)
    for raw_pred, pred, gt, im, r in zip(raw_preds, sim_preds, cpu_texts_decode, cpu_images, cpu_rounds):
        im = ((im.numpy() * 0.5) + 0.5) * 255
        im = np.transpose(im, (1, 2, 0))
        filename = '{} - {} - {}.jpeg'.format(r, gt.replace(',', ' '), pred.replace(',', ' '))
        filename = filename.replace(':', '').replace('ú', 'u').replace('ö', 'o').replace('!', '')
        cv2.imwrite(os.path.join(example_dir, filename), im)
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    accuracy = n_correct / n_total
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


def val_ocr(net, val_loader, criterion, working_dir, best_val_loss, image, text, length):
    print('Start val')
    n_test_disp = 10
    for p in net.parameters():
        p.requires_grad = False

    net.eval()

    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    n_total = 0
    loss_avg = Averager()
    with torch.no_grad():
        for index in range(len(val_loader)):
            data = val_iter.next()

            d = data
            inputs = d[0]
            outputs = d[1]
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

            cost = criterion(preds, text, preds_size, length)
            loss_avg.add(cost)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds = preds.to('cpu')
            sim_preds = net.converter.decode(preds.data, preds_size.data, raw=False)
            cpu_texts_decode = net.converter.decode(cpu_texts, cpu_lengths, raw=True)
            for i, (pred, target) in enumerate(zip(sim_preds, cpu_texts_decode)):
                if pred == target:
                    n_correct += 1
            n_total += batch_size

    raw_preds = net.converter.decode(preds.data, preds_size.data, raw=True)[:n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / n_total
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
