import torch
import os
import torch.nn.functional as F
from annotator.training.helper import Averager


def train_batch(net, train_iter, device, losses, optimizer):
    #cnn_encoder, rnn_decoder = net
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
    #predicteds = rnn_decoder(cnn_encoder(inputs))
    predicteds = net(inputs)
    loss = None

    for k, v in predicteds.items():
        v = v.view(v.size(0) * v.size(1), -1)
        labs = labels[k].view(labels[k].size(0) * labels[k].size(1))
        if loss is None:
            #loss = losses[k](v, labels[k])
            loss = losses[k](v, labs)
        else:
            loss += losses[k](v, labs)
    loss.backward()
    optimizer.step()
    return loss


def val(net, val_loader, device, losses, working_dir, best_val_loss):
    #cnn_encoder, rnn_decoder = net
    rnn_decoder = net
    print('Start val')

    #for p in cnn_encoder.parameters():
    #    p.requires_grad = False

    #cnn_encoder.eval()

    for p in rnn_decoder.parameters():
        p.requires_grad = False
    rnn_decoder.eval()

    val_iter = iter(val_loader)
    i = 0
    n_correct = 0
    loss_avg = Averager()
    corrects = {k: 0 for k in list(rnn_decoder.sets.keys())}

    total = 0
    label_corrects = {}
    label_totals = {}
    for k in corrects.keys():
        label_corrects[k] = list(0. for i in range(len(rnn_decoder.sets[k])))
        label_totals[k] = list(0. for i in range(len(rnn_decoder.sets[k])))

    for index in range(len(val_loader)):
        data = val_iter.next()

        inputs, labels = data
        for k, v in inputs.items():
            inputs[k] = v[0].float().to(device)
        for k, v in labels.items():
            labels[k] = v[0].long().to(device)

        #predicteds = rnn_decoder(cnn_encoder(inputs))
        predicteds = rnn_decoder(inputs)

        loss = None

        for k, v in predicteds.items():
            v = v.view(v.size(0) * v.size(1), -1)
            labs = labels[k].view(labels[k].size(0) * labels[k].size(1))
            if loss is None:
                #loss = losses[k](v, labels[k])
                loss = losses[k](v, labs)
            else:
                loss += losses[k](v, labs)
        loss_avg.add(loss)
        for k, v in predicteds.items():
            for i in range(v.size(0)):
                _, predicteds[k] = torch.max(v[i], 1)
                labs = labels[k][i]
                corrects[k] += (predicteds[k] == labs).sum().item()
                c = (predicteds[k] == labs).squeeze().to('cpu')
                if c.shape:
                    for i in range(c.shape[0]):
                        label = labs[i]
                        label_corrects[k][label] += c[i].item()
                        label_totals[k][label] += 1
                else:
                    label = labs[0]
                    label_corrects[k][label] += c.item()
                    label_totals[k][label] += 1

        total += inputs['image'].size(0) * inputs['image'].size(1)
    print('Val loss: {}'.format(loss_avg.val()))
    for k, v in corrects.items():
        print('%s accuracy of the network on the %d test images: %d %%' % (k, total,
            100 * corrects[k] / total))

    if loss_avg.val() < best_val_loss:
        print('Saving new best model!')
        # torch.save(cnn_encoder.state_dict(), os.path.join(working_dir, 'cnn_model.pth'))
        torch.save(rnn_decoder.state_dict(), os.path.join(working_dir, 'model.pth'))
        best_val_loss = loss_avg.val()

    assessment_file = os.path.join(working_dir, 'accuracy.txt')

    import datetime

    with open(assessment_file, 'a', encoding='utf8') as f:
        f.write('\n\n' + str(datetime.datetime.now()) + '\n')
        for k in corrects.keys():
            print(k)
            for i in range(len(rnn_decoder.sets[k])):
                if not label_totals[k][i]:
                    continue
                print('Accuracy of %5s : %2d %% (%d / %d)' % (
                    rnn_decoder.sets[k][i], 100 * label_corrects[k][i] / label_totals[k][i], label_corrects[k][i], label_totals[k][i]))
                print('Accuracy of %5s : %2d %% (%d / %d)' % (
                    rnn_decoder.sets[k][i], 100 * label_corrects[k][i] / label_totals[k][i], label_corrects[k][i], label_totals[k][i]), file=f)
    return best_val_loss
