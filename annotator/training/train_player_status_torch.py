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

working_dir = r'E:\Data\Overwatch\models\player_status'
os.makedirs(working_dir, exist_ok=True)
log_dir = os.path.join(working_dir, 'log')
TEST = True
train_dir = r'E:\Data\Overwatch\training_data\player_status'
hdf5_path = os.path.join(train_dir, 'dataset.hdf5')

cuda = True
seed = 1
batch_size = 12
test_batch_size = 100
epochs = 10
lr = 0.01
momentum = 0.5
log_interval = 10

set_files = {#'player': os.path.join(train_dir, 'player_set.txt'),
             'hero': os.path.join(train_dir, 'hero_set.txt'),
             'alive': os.path.join(train_dir, 'alive_set.txt'),
             'ult': os.path.join(train_dir, 'ult_set.txt'),
             #'antiheal': os.path.join(train_dir, 'antiheal_set.txt'),
             #'asleep': os.path.join(train_dir, 'asleep_set.txt'),
             #'frozen': os.path.join(train_dir, 'frozen_set.txt'),
             #'hacked': os.path.join(train_dir, 'hacked_set.txt'),
             #'stunned': os.path.join(train_dir, 'stunned_set.txt'),
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
                if count > 10:
                    break

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
        weights = {}
        for k in sets.keys():
            total = np.sum(np.array(list(counters[k].values())))
            for k2, v2 in counters[k].items():
                print(sets[k][k2], v2)
            w = np.zeros((len(sets[k]),))
            for k2, v in counters[k].items():
                score = total/float(v)
                w[k2] = score
            weights[k] = torch.from_numpy(w).float()
        print('DONE SETTING UP WEIGHTS')
        return weights

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
            inputs['image']= torch.from_numpy(np.moveaxis(hf5['train_img'][real_index:real_index+batch_size, ...], -1, 2)).float()
            for k in sets.keys():
                outputs[k] = torch.from_numpy(hf5['train_{}_label'.format(k)][real_index:real_index+batch_size]).long()
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
                if count > 10:
                    break

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
            for k in sets.keys():
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


hero_classes = sets['hero']
ult_classes = sets['ult']
alive_classes = sets['alive']

def imshow(img):
    img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))[:,:, [2,1,0]]/255)
    plt.show()

# get some random training images
dataiter = iter(trainloader)
inputs, outputs = dataiter.next()
print(inputs['image'].shape)
print(outputs['hero'].shape)
# show images
imshow(torchvision.utils.make_grid(inputs['image'][0, :4, 0, ...]))
# print labels
for k in sets.keys():
    print(k, ' '.join('%5s' % sets[k][int(outputs[k][0, j, 0])] for j in range(4)))

### TUTORIAL CODE


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.outputs = {}
        for k, v in sets.items():
            self.outputs[k] = nn.Linear(84, len(v))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_outs = {}
        for k, v in self.outputs.items():
            x_outs[k] = v(x)
        return x_outs


class EmbeddedCNN(nn.Module):
    def __init__(self):
        super(EmbeddedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        return x

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class Combine(nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.cnn = EmbeddedCNN()
        self.rnn = nn.LSTM(
            input_size=16 * 13 * 13,
            hidden_size=64,
            num_layers=3,
            batch_first=True)
        self.spectator_mode_input = nn.Linear(len(spectator_modes), 10)

        for k, v in sets.items():
            setattr(self, '{}_output'.format(k), TimeDistributed(nn.Linear(64, len(v)),batch_first=True))

    def forward(self, x):
        x = x['image']
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)

        x_outs = {}
        for k in sets.keys():
            x_outs[k] = F.log_softmax(getattr(self,'{}_output'.format(k))(r_out), dim=2)

        return x_outs

net = Combine()
net.to(device)


class Loss(nn.CrossEntropyLoss):
    def forward(self, input, target):
        batch_size, time_steps, classes = input.shape
        i = input.view(batch_size * time_steps, classes)
        t = target.view(batch_size * time_steps)
        return F.cross_entropy(i, t, weight=self.weight,
                            ignore_index=self.ignore_index, reduction=self.reduction)

weights = train_set.generate_class_weights(mu=10)
print('WEIGHTS')
for k, v in weights.items():
    print(k)
    print(', '.join('{}: {}'.format(sets[k][k2],v2) for k2, v2 in enumerate(v)))


losses = {}
for k in sets.keys():
    losses[k] = Loss(weight=weights[k])
    losses[k].to(device)

optimizer = optim.Adadelta(net.parameters())
import time
for epoch in range(2):  # loop over the dataset multiple times
    batch_begin = time.time()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        corrects = {k: 0 for k in sets.keys()}
        print(i)
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
            if k != 'hero':
                continue
            if loss is None:
                loss = losses[k](v, labels[k])
            else:
                loss += losses[k](v, labels[k])
        for k, v in predicteds.items():
            _, predicteds[k] = torch.max(v, 2)
            corrects[k] += (predicteds[k] == labels[k]).sum().item()
        #print('Loss calculation took: {}'.format(time.time()-begin))
        #begin = time.time()
        loss.backward()
        optimizer.step()
        #print('Back prop took: {}'.format(time.time()-begin))

        # print statistics
        #begin = time.time()
        running_loss += loss.item()
        if i % 20 == 19 or i != 0:    # print every 2000 mini-batches
            print('Epoch %d, %d/%d, loss: %.3f' %
                  (epoch + 1, i + 1, len(train_set), running_loss / i))
            print(', '.join('{} accuracy={}'.format(k,v/(inputs['image'].shape[0] * inputs['image'].shape[1])) for k, v in corrects.items()))
            running_loss = 0.0
        #print('Running loss calc took: {}'.format(time.time()-begin))
        print('Batch took: {}'.format(time.time()-batch_begin))
        batch_begin = time.time()


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
imshow(torchvision.utils.make_grid(inputs['image'][:4, 0, ...]))
predicteds = net(inputs)
for k in sets.keys():
    print(k, 'GroundTruth: ', ' '.join('%5s' % sets[k][int(labels[k][j, 0])] for j in range(4)))

    _, predicteds[k] = torch.max(predicteds[k] , 2)
    print('Predicted {}: '.format(k), ' '.join('%5s' % sets[k][int(predicteds[k][j, 0])]
                              for j in range(4)))

corrects = {k: 0 for k in sets.keys()}
hero_correct = list(0. for i in range(len(sets['hero'])))
hero_total = list(0. for i in range(len(sets['hero'])))

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
            _, predicteds[k] = torch.max(v, 2)
            corrects[k] += (predicteds[k] == labels[k]).sum().item()
            if k == 'hero':
                c = (predicteds[k] == labels[k]).squeeze()
                for i in range(predicteds[k].shape[0]):
                    for j in range(predicteds[k].shape[1]):
                        label = labels[k][i, j]
                        hero_correct[label] += c[i, j].item()
                        hero_total[label] += 1

        total += inputs['image'].size(0) * inputs['image'].size(1)
        print('Test batch took: {}'.format(time.time()-batch_begin))
        batch_begin = time.time()

for k, v in corrects.items():
    print('Hero accuracy of the network on the 10000 test images: %d %%' % (
        100 * corrects[k] / total))

for i in range(len(hero_classes)):
    if not hero_total[i]:
        continue
    print('Accuracy of %5s : %2d %%' % (
        hero_classes[i], 100 * hero_correct[i] / hero_total[i]))

