import torch
import os
from annotator.training.helper import Averager


def train_batch(net, train_iter, device, losses, optimizer):
    #import time
    #begin = time.time()
    data = train_iter.next()
    #print('Data loading took', time.time()-begin)

    inputs, labels = data
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
    for k, v in corrects.items():
        print('%s accuracy of the network on the %d test images: %d %%' % (k, total,
            100 * corrects[k] / total))

    if loss_avg.val() < best_val_loss and not fine_tune:
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
