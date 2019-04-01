import os
import torch
import torchvision
import torchvision.transforms as transforms
import h5py
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import random

working_dir = r'E:\Data\Overwatch\models\player_status'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')
TEST = True
train_dir = r'E:\Data\Overwatch\training_data\player_status'
hdf5_path = os.path.join(train_dir, 'dataset.hdf5')

cuda = True
seed = 1
batch_size = 100
test_batch_size = 100
epochs = 10
lr = 0.01
momentum = 0.5
log_interval = 10

set_files = {#'player': os.path.join(train_dir, 'player_set.txt'),
             'hero': os.path.join(train_dir, 'hero_set.txt'),
             'alive': os.path.join(train_dir, 'alive_set.txt'),
             'ult': os.path.join(train_dir, 'ult_set.txt'),
             'antiheal': os.path.join(train_dir, 'antiheal_set.txt'),
             'asleep': os.path.join(train_dir, 'asleep_set.txt'),
             'frozen': os.path.join(train_dir, 'frozen_set.txt'),
             'hacked': os.path.join(train_dir, 'hacked_set.txt'),
             'stunned': os.path.join(train_dir, 'stunned_set.txt'),
             #'spectator': os.path.join(train_dir, 'spectator_set.txt'),
             }
end_set_files = {
           # 'color': os.path.join(train_dir, 'color_set.txt'),
}

def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


sets = {}
for k, v in set_files.items():
    sets[k] = load_set(v)
end_sets = {}
for k, v in end_set_files.items():
    end_sets[k] = load_set(v)

class_counts = {}
for k, v in sets.items():
    class_counts[k] = len(v)

end_class_counts = {}
for k, v in end_sets.items():
    end_class_counts[k] = len(v)

sides = load_set(os.path.join(train_dir, 'side_set.txt'))

side_count = len(sides)

colors = load_set(os.path.join(train_dir, 'color_set.txt'))

color_count = len(colors)

spectator_modes = load_set(os.path.join(train_dir, 'spectator_mode_set.txt'))

spectator_mode_count = len(spectator_modes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class TrainDataset(data.Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        self.data_num = 0
        self.data_indices = {}
        count = 0
        for f in os.listdir(train_dir):
            if f.endswith('.hdf5'):
                with h5py.File(os.path.join(train_dir, f), 'r') as h5f:
                    self.data_num += h5f['train_img'].shape[0]
                    self.data_indices[self.data_num] = os.path.join(train_dir, f)
                count += 1
                #if count > 1:
                #    break
        self.weights = {}
        print('DONE SETTING UP')

    def __len__(self):
        return int(self.data_num / batch_size)

    def generate_class_weights(self, mu=0.5):
        from collections import Counter

        counters = {}
        weights = {}
        for k, v in class_counts.items():
            print(k, v)
            counters[k] = Counter()
        end_counters = {}
        for k, v in end_class_counts.items():
            print(k, v)
            end_counters[k] = Counter()
        for i, (next_ind, path) in enumerate(self.data_indices.items()):
            with h5py.File(path, 'r') as hf5:

                for k, v in class_counts.items():
                    y_train = hf5['train_{}_label'.format(k)]
                    unique, counts = np.unique(y_train, return_counts=True)
                    counts = dict(zip(unique, counts))
                    counters[k].update(counts)

                for k, v in end_class_counts.items():
                    y_train = hf5['train_{}_label'.format(k)]
                    unique, counts = np.unique(y_train, return_counts=True)
                    counts = dict(zip(unique, counts))
                    end_counters[k].update(counts)
                #print(i, path, start_ind, next_ind, next_ind - start_ind, hf5['train_img'].shape)
        self.weights = {}
        for k in sets.keys():
            total = np.sum(np.array(list(counters[k].values())))
            for k2, v2 in counters[k].items():
                print(sets[k][k2], v2)
            w = np.zeros((len(sets[k]),))
            for k2, v in counters[k].items():
                score = total/float(v)
                w[k2] = score if score <= 1000 else 1000
            self.weights[k] = torch.from_numpy(w).float()
        for k in end_sets.keys():
            total = np.sum(np.array(list(end_counters[k].values())))
            for k2, v2 in end_counters[k].items():
                print(end_sets[k][k2], v2)
            w = np.zeros((len(end_sets[k]),))
            for k2, v in end_counters[k].items():
                score = total/float(v)
                w[k2] = score if score <= 1000 else 1000
            self.weights[k] = torch.from_numpy(w).float()
        print('DONE SETTING UP WEIGHTS')
        return self.weights

    def __getitem__(self, index):
        start_ind = 0
        real_index = index * batch_size
        for i, (next_ind, v) in enumerate(self.data_indices.items()):
            path = v
            if real_index < next_ind:
                break
            start_ind = next_ind

        real_index = real_index - start_ind
        inputs = {}
        outputs = {}
        with h5py.File(path, 'r') as hf5:

            for k in sets.keys():
                outputs[k] = torch.from_numpy(hf5['train_{}_label'.format(k)][real_index:real_index+batch_size]).long()
                b, time_steps= outputs[k].shape
                outputs[k] = outputs[k].view(b*time_steps)

            for k in end_sets.keys():
                outputs[k] = torch.from_numpy(hf5['train_{}_label'.format(k)][real_index:real_index+batch_size]).long()
            #ind = random.randint(0,99)

            inputs['image']= torch.from_numpy(np.moveaxis(hf5['train_img'][real_index:real_index+batch_size, ...], -1, 2)).float()
            b, time_steps, channels, H, W = inputs['image'].shape
            inputs['image'] = inputs['image'].view(b * time_steps, channels, H, W)
        return inputs, outputs


class TestDataset(data.Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        self.data_num = 0
        self.data_indices = {}
        count=0
        for f in os.listdir(train_dir):
            if f.endswith('.hdf5'):
                with h5py.File(os.path.join(train_dir, f), 'r') as h5f:
                    self.data_num += h5f['val_img'].shape[0]
                    self.data_indices[self.data_num] = os.path.join(train_dir, f)
                count += 1
                #if count > 1:
                #    break

    def __getitem__(self, index):
        start_ind = 0
        real_index = index * test_batch_size
        for i, (next_ind, v) in enumerate(self.data_indices.items()):
            path = v
            if real_index < next_ind:
                break
            start_ind = next_ind
        real_index = real_index - start_ind
        inputs = {}
        outputs = {}
        with h5py.File(path, 'r') as hf5:
            inputs['image']= torch.from_numpy(np.moveaxis(hf5['val_img'][real_index:real_index+test_batch_size, ...], -1, 2)).float()
            b, time_steps, channels, H, W = inputs['image'].shape
            inputs['image'] = inputs['image'].view(b * time_steps, channels, H, W)
            for k in sets.keys():
                outputs[k] = torch.from_numpy(hf5['val_{}_label'.format(k)][real_index:real_index+test_batch_size]).long()
                b, time_steps= outputs[k].shape
                outputs[k] = outputs[k].view(b*time_steps)

            for k in end_sets.keys():
                outputs[k] = torch.from_numpy(hf5['val_{}_label'.format(k)][real_index:real_index+test_batch_size]).long()
        return inputs, outputs

    def __len__(self):
        return int(self.data_num / test_batch_size)


train_set = TrainDataset()
trainloader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                          shuffle=True)
test_set = TestDataset()
testloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=True)

def imshow(img):
    img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))[:,:, [2,1,0]])
    plt.show()

# get some random training images
dataiter = iter(trainloader)
inputs, outputs = dataiter.next()
print(inputs['image'].shape)
print(outputs['hero'].shape)
# show images
imshow(torchvision.utils.make_grid(inputs['image'][0, :4, ...]))
# print labels
for k in sets.keys():
    print(k, ' '.join('%5s' % sets[k][int(outputs[k][0, j])] for j in range(4)))

### TUTORIAL CODE


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.fc1 = nn.Linear(10 * 13 * 13, 84)
        for k, v in sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(84, len(v)))
        for k, v in end_sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(84, len(v)))

    def forward(self, x):
        x = x['image']
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 13 * 13)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        for k in end_sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs

net = CNN()
net.to(device)

weights = train_set.generate_class_weights(mu=10)
print('WEIGHTS')
for k, v in weights.items():
    print(k)
    if k in sets:
        print(', '.join('{}: {}'.format(sets[k][k2],v2) for k2, v2 in enumerate(v)))
    else:
        print(', '.join('{}: {}'.format(end_sets[k][k2],v2) for k2, v2 in enumerate(v)))

losses = {}
for k in sets.keys():
    #losses[k] = Loss(weight=weights[k])
    losses[k] = nn.CrossEntropyLoss(weight=weights[k])
    losses[k].to(device)
for k in end_sets.keys():
    #losses[k] = Loss(weight=weights[k])
    losses[k] = nn.CrossEntropyLoss(weight=weights[k])
    losses[k].to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.09)
import time
for epoch in range(5):  # loop over the dataset multiple times
    batch_begin = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        corrects = {k: 0 for k in list(sets.keys()) + list(end_sets.keys())}
        # get the inputs
        #begin = time.time()
        inputs, labels = data
        for k,v in inputs.items():
            v = v[0]
            inputs[k] = v.to(device)
        for k,v in labels.items():
            v = v[0]
            labels[k] = v.to(device)
        #print('Loading data took: {}'.format(time.time()-begin))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #begin = time.time()
        predicteds = net(inputs)
        #print('Predicting data took: {}'.format(time.time()-begin))
        loss = None
        #begin = time.time()
        for k, v in predicteds.items():
            if loss is None:
                loss = losses[k](v, labels[k])
            else:
                loss += losses[k](v, labels[k])
        for k, v in predicteds.items():
            _, predicteds[k] = torch.max(v, 1)
            corrects[k] += (predicteds[k] == labels[k]).sum().item()
        #print('Loss calculation took: {}'.format(time.time()-begin))
        #begin = time.time()
        loss.backward()
        optimizer.step()
        #print('Back prop took: {}'.format(time.time()-begin))

        # print statistics
        #begin = time.time()
        running_loss += loss.item()
        if i % 50 == 49:    # print every 2000 mini-batches
            #print(predicteds['hero'])
            #print(labels['hero'])
            print('Epoch %d, %d/%d, loss: %.3f' %
                  (epoch + 1, i + 1, len(train_set), running_loss / i))
            print(', '.join('{} accuracy={}'.format(k,v/(inputs['image'].shape[0])) for k, v in corrects.items()))
            running_loss = 0.0
            print('Batch took: {}'.format(time.time()-batch_begin))
        batch_begin = time.time()
        #print('Running loss calc took: {}'.format(time.time()-begin))


print('Finished Training')


dataiter = iter(testloader)
inputs, labels = dataiter.next()
for k,v in inputs.items():
    v = v[0]
    inputs[k] = v.to(device)
for k,v in labels.items():
    v = v[0]
    labels[k] = v.to(device)

# print images
imshow(torchvision.utils.make_grid(inputs['image'][:4, ...]))
predicteds = net(inputs)
for k in sets.keys():
    print(labels[k].shape)
    print(k, 'GroundTruth: ', ' '.join('%5s' % sets[k][int(labels[k][j])] for j in range(4)))

    _, predicteds[k] = torch.max(predicteds[k] ,1)
    print('Predicted {}: '.format(k), ' '.join('%5s' % sets[k][int(predicteds[k][j])]
                              for j in range(4)))

corrects = {k: 0 for k in list(sets.keys()) + list(end_sets.keys())}
label_corrects = {}
label_totals = {}
for k in corrects.keys():
    if k in sets:
        label_corrects[k] = list(0. for i in range(len(sets[k])))
        label_totals[k] = list(0. for i in range(len(sets[k])))
    else:
        label_corrects[k] = list(0. for i in range(len(end_sets[k])))
        label_totals[k] = list(0. for i in range(len(end_sets[k])))

total = 0
with torch.no_grad():
    batch_begin = time.time()
    for i, data in enumerate(testloader):
        print(i, len(testloader))
        inputs, labels = data
        for k,v in inputs.items():
            v = v[0]
            inputs[k] = v.to(device)
        for k,v in labels.items():
            v = v[0]
            labels[k] = v.to(device)

        predicteds = net(inputs)
        for k, v in predicteds.items():
            _, predicteds[k] = torch.max(v, 1)
            corrects[k] += (predicteds[k] == labels[k]).sum().item()
            c = (predicteds[k] == labels[k]).squeeze()
            for i in range(c.shape[0]):
                label = labels[k][i]
                label_corrects[k][label] += c[i].item()
                label_totals[k][label] += 1

        total += inputs['image'].size(0)
        print('Test batch took: {}'.format(time.time()-batch_begin))
        batch_begin = time.time()

for k, v in corrects.items():
    print('%s accuracy of the network on the %d test images: %d %%' % (k, total,
        100 * corrects[k] / total))

model_file = os.path.join(working_dir, 'player_status.model')
with open(model_file, 'wb') as f:
    torch.save(net, f)

assessment_file = os.path.join(working_dir, 'accuracy.txt')

import datetime

with open(assessment_file, 'a', encoding='utf8') as f:
    f.write('\n\n' + str(datetime.datetime.now()) + '\n')
    for k in corrects.keys():
        print(k)
        if k in sets:
            for i in range(len(sets[k])):
                if not label_totals[k][i]:
                    continue
                print('Accuracy of %5s : %2d %% (%d / %d)' % (
                    sets[k][i], 100 * label_corrects[k][i] / label_totals[k][i], label_corrects[k][i], label_totals[k][i]))
                print('Accuracy of %5s : %2d %% (%d / %d)' % (
                    sets[k][i], 100 * label_corrects[k][i] / label_totals[k][i], label_corrects[k][i], label_totals[k][i]), file=f)
        else:
            for i in range(len(end_sets[k])):
                if not label_totals[k][i]:
                    continue
                print('Accuracy of %5s : %2d %% (%d / %d)' % (
                    end_sets[k][i], 100 * label_corrects[k][i] / label_totals[k][i], label_corrects[k][i], label_totals[k][i]))
                print('Accuracy of %5s : %2d %% (%d / %d)' % (
                    end_sets[k][i], 100 * label_corrects[k][i] / label_totals[k][i], label_corrects[k][i], label_totals[k][i]), file=f)

