import torch
import os
import shutil
import cv2
from annotator.training.helper import Averager, load_set, lr, beta1, display_interval, print_params, \
    set_train, load_checkpoint, weights_init
import torch.nn as nn
import torch.optim as optim
import numpy as np



def train_model(working_dir, train_dir, model_class, data_class, set_files, spec_modes=None,
                    input_set_files=None, early_stopping_threshold=2, batch_size=100, test_batch_size=200, num_epochs=50):
    base_model_path = os.path.join(working_dir, 'model.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sets = {}
    for k, v in set_files.items():
        shutil.copyfile(v, os.path.join(working_dir, '{}_set.txt'.format(k)))
        sets[k] = load_set(v)
    input_sets = {}
    if input_set_files:
        for k, v in input_set_files.items():
            shutil.copyfile(v, os.path.join(working_dir, '{}_set.txt'.format(k)))
            input_sets[k] = load_set(v)
    if spec_modes:
        train_set = data_class(train_dir, sets=sets, input_sets=input_sets, batch_size=batch_size,
                                           pre='train', modes=spec_modes)
        test_set = data_class(train_dir, sets=sets, input_sets=input_sets, batch_size=test_batch_size,
                                          pre='val', modes=spec_modes)
    else:
        train_set = data_class(train_dir, sets=sets, input_sets=input_sets, batch_size=batch_size,
                                           pre='train')
        test_set = data_class(train_dir, sets=sets, input_sets=input_sets, batch_size=test_batch_size,
                                          pre='val')

    net = model_class(sets, input_sets)

    losses = {}
    for k in sets.keys():
        losses[k] = nn.CrossEntropyLoss()  # weight=weights[k])
        losses[k].to(device)

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

            cost = train_batch(net, train_iter, device, losses, optimizer)
            loss_avg.add(cost)
            i += 1

            if i % display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        prev_best = best_val_loss
        with open(loss_path, 'a') as f:
            f.write('Epoch {}: '.format(epoch))
        best_val_loss = val(net, val_loader, device, losses, working_dir, best_val_loss)
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


def test_errors(model_dir, working_dir, train_dir, model_class, data_class, set_files, spec_mode=None,
                input_set_files=None, batch_size=600):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Processing ' + spec_mode)

    sets = {}
    for k, v in set_files.items():
        sets[k] = load_set(v)
    input_sets = {}
    if input_set_files:
        for k, v in input_set_files.items():
            input_sets[k] = load_set(v)
    net = model_class(sets, input_sets)
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
            test_set = data_class(train_dir, sets=sets, input_sets=input_sets, batch_size=batch_size, pre=pre,
                                              modes=[spec_mode])
        else:
            test_set = data_class(train_dir, sets=sets, input_sets=input_sets, batch_size=batch_size, pre=pre)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                 shuffle=False, num_workers=4, pin_memory=True)
        print(len(val_loader), 'batches')

        val_iter = iter(val_loader)
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
                    if k in ['round', 'vod', 'time_point']:
                        inputs[k] = v[0]
                        continue
                    inputs[k] = v[0].float().to(device)
                for k, v in labels.items():
                    labels[k] = v[0].long().to(device)
                predicteds = net(inputs)

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
                            try:
                                r = inputs['round'][i].item()
                            except KeyError:
                                r = inputs['vod'][i].item()
                            if c[i].item() == 0:
                                errors[k].append({'image': inputs['image'][i, ...].cpu().numpy(),
                                                  'truth': net.sets[k][label],
                                                  'predicted': net.sets[k][predicteds[k][i]],
                                                  'round': r,
                                                  'time_point': inputs['time_point'][i].item()})
                    else:
                        label = labels[k][0]
                        label_corrects[k][label] += c.item()
                        label_totals[k][label] += 1
                total += inputs['image'].size(0)
                for k, e in errors.items():
                    for error in e:
                        im = ((np.transpose(error['image'], (1, 2, 0)) * 0.5) + 0.5) * 255
                        directory = os.path.join(working_dir, k,
                                                 error['truth'].replace(':', '').replace(':', '').replace('ú',
                                                                                                          'u').replace(
                                                     'ö', 'o').replace('/', ''))
                        if not os.path.exists(directory):
                            os.makedirs(directory, exist_ok=True)
                        minutes = int(error['time_point'] / 60)
                        seconds = round(error['time_point'] - int(minutes * 60), 1)
                        filename = '{} - {}_{} - predicted_{}.jpeg'.format(error['round'],
                                                                           minutes, seconds,
                                                                           error['predicted']).replace(':', '').replace(
                            'ú', 'u').replace('ö', 'o').replace('/', '')
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
                        net.sets[k][i], 100 * label_corrects[k][i] / label_totals[k][i], label_corrects[k][i],
                        label_totals[k][i]))
                    print('Accuracy of %5s : %2d %% (%d / %d)' % (
                        net.sets[k][i], 100 * label_corrects[k][i] / label_totals[k][i], label_corrects[k][i],
                        label_totals[k][i]), file=f)


def fine_tune_model(working_dir, train_dir, model_class, data_class, set_files, spec_mode,
                    input_set_files=None, early_stopping_threshold=2, batch_size=100, test_batch_size=200, num_epochs=50):

    base_model_path = os.path.join(working_dir, 'model.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ft_working_dir = os.path.join(working_dir, spec_mode)
    if os.path.exists(ft_working_dir):
        return
    os.makedirs(ft_working_dir, exist_ok=True)

    sets = {}
    for k, v in set_files.items():
        shutil.copyfile(v, os.path.join(ft_working_dir, '{}_set.txt'.format(k)))
        sets[k] = load_set(v)
    input_sets = {}
    if input_set_files:
        for k, v in input_set_files.items():
            shutil.copyfile(v, os.path.join(ft_working_dir, '{}_set.txt'.format(k)))
            input_sets[k] = load_set(v)

    train_set = data_class(train_dir, sets=sets, input_sets=input_sets, batch_size=batch_size,
                                       pre='train', modes=[spec_mode])
    test_set = data_class(train_dir, sets=sets, input_sets=input_sets, batch_size=test_batch_size,
                                      pre='val', modes=[spec_mode])

    print(len(train_set))
    net = model_class(sets, input_sets)

    if os.path.exists(base_model_path):  # Initialize from CNN model
        d = torch.load(base_model_path)
        net.load_state_dict(d, strict=False)
        print('Initialized from base model!')

    losses = {}
    for k in sets.keys():
        losses[k] = nn.CrossEntropyLoss()  # weight=weights[k])
        losses[k].to(device)

    net = set_train(net, True, fine_tune=True)
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
        best_val_loss = val(net, val_loader, device, losses, ft_working_dir, best_val_loss)
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

            cost = train_batch(net, train_iter, device, losses, optimizer)
            loss_avg.add(cost)
            i += 1

            if i % display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        prev_best = best_val_loss
        with open(loss_path, 'a') as f:
            f.write('Epoch {}: '.format(epoch))
        best_val_loss = val(net, val_loader, device, losses, ft_working_dir, best_val_loss)
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

def train_batch(net, train_iter, device, losses, optimizer):
    #import time
    #begin = time.time()
    data = train_iter.next()
    #print('Data loading took', time.time()-begin)

    inputs, labels = data
    batch_size = inputs['image'].size(0)
    if not batch_size:
        return
    for k, v in inputs.items():
        inputs[k] = v[0].to(device)
    for k, v in labels.items():
        labels[k] = v[0].long().to(device)
    optimizer.zero_grad()
    predicteds = net(inputs)

    loss = None

    for k, v in predicteds.items():
        if loss is None:
            loss = losses[k](v, labels[k])
        else:
            loss += losses[k](v, labels[k])
    loss.backward()
    optimizer.step()
    return loss


def val(net, val_loader, device, losses, working_dir, best_val_loss, fine_tune=False):
    print('Start val')

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
            data = val_iter.next()

            inputs, labels = data
            batch_size = inputs['image'].size(0)
            if not batch_size:
                continue
            for k, v in inputs.items():
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
            for k, v in predicteds.items():
                _, predicteds[k] = torch.max(v, 1)
                corrects[k] += (predicteds[k] == labels[k]).sum().item()
                c = (predicteds[k] == labels[k]).squeeze().to('cpu')
                if c.shape:
                    for i in range(c.shape[0]):
                        label = labels[k][i]
                        label_corrects[k][label] += c[i].item()
                        label_totals[k][label] += 1
                else:
                    label = labels[k][0]
                    label_corrects[k][label] += c.item()
                    label_totals[k][label] += 1

            total += inputs['image'].size(0)
    print('Val loss: {}'.format(loss_avg.val()))
    with open(os.path.join(working_dir, 'losses.txt'), 'a') as f:
        f.write('Val loss: {}\n'.format(loss_avg.val()))
    for k, v in corrects.items():
        print('%s accuracy of the network on the %d test images: %d %%' % (k, total,
            100 * corrects[k] / total))

    if loss_avg.val() < best_val_loss and not fine_tune:
        with open(os.path.join(working_dir, 'losses.txt'), 'a') as f:
            f.write('Saving new best model!\n')
        print('Saving new best model!')
        torch.save(net.state_dict(), os.path.join(working_dir, 'model.pth'))
        best_val_loss = loss_avg.val()
    elif loss_avg.val() < best_val_loss and fine_tune:
        print('Saving new best model!')
        torch.save(net.state_dict(), os.path.join(working_dir, 'model_fine_tuned.pth'))
        best_val_loss = loss_avg.val()

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
    return best_val_loss
