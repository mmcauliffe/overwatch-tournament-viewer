import torch
import torch.nn as nn
import torch.nn.functional as F


class StatusCNN(nn.Module):
    def __init__(self, sets, input_sets):
        super(StatusCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5)
        self.fc1 = nn.Linear(10 * 13 * 13, 84)
        self.sets = sets
        self.input_sets = input_sets
        self.embeddings = {}
        input_length = 0
        for k, v in self.input_sets.items():
            setattr(self, '{}_input'.format(k), nn.Embedding(len(v), len(v)))
            input_length += len(v)
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(84 + input_length, len(v)))

    def forward(self, x):
        inputs = {}
        for k, v in self.input_sets.items():
            inputs[k] = x[k]
        x = x['image']
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 13 * 13)

        x = F.relu(self.fc1(x))
        for k, v in inputs.items():
            extra_input = getattr(self,'{}_input'.format(k))(v.to(torch.int64))
            x = torch.cat((x, extra_input), 1)
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs

class MidCNN(nn.Module):
    def __init__(self, sets):
        super(MidCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5)
        self.fc1 = nn.Linear(10 * 9 * 33, 128)
        self.sets = sets
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(128, len(v)))

    def forward(self, x):
        x = x['image']
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 9 * 33)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs

class PauseCNN(nn.Module):
    def __init__(self, sets):
        super(PauseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5)
        self.fc1 = nn.Linear(10 * 3 * 15, 128)
        self.sets = sets
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(128, len(v)))

    def forward(self, x):
        x = x['image']
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 3 * 15)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs

class ReplayCNN(nn.Module):
    def __init__(self, sets):
        super(ReplayCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5)
        self.fc1 = nn.Linear(10 * 5 * 23, 128)
        self.sets = sets
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(128, len(v)))

    def forward(self, x):
        x = x['image']
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 23)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs

class KillFeedCNN(nn.Module):
    def __init__(self, sets):
        super(KillFeedCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5)
        self.fc1 = nn.Linear(10 * 5 * 59, 128)
        self.sets = sets
        for k, v in self.sets.items():
            setattr(self, '{}_output'.format(k), nn.Linear(128, len(v)))

    def forward(self, x):
        x = x['image']
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 5 * 59)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x_outs = {}
        for k in self.sets.keys():
            x_outs[k] = getattr(self,'{}_output'.format(k))(x)
        return x_outs
