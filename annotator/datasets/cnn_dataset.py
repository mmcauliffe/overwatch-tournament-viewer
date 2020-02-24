from torch.utils.data import Dataset
import torch
import os
import h5py
import pickle
import sys
import cv2
from collections import Counter
import numpy as np



class CNNHDF5Dataset(Dataset):
    def __init__(self, train_dir, sets, input_sets=None, batch_size=100, pre='train', modes=None):
        if modes is None:
            modes = ['original']
        self.batch_size = batch_size
        self.pre = pre
        self.sets = sets
        if input_sets is None:
            input_sets = {}
        self.input_sets = input_sets
        self.class_counts = {}
        for k, v in sets.items():
            self.class_counts[k] = len(v)
        self.data_num = 0
        self.data_indices = {}
        count = 0
        for m in modes:
            m_dir = os.path.join(train_dir, m)
            for f in os.listdir(m_dir):
                if f.endswith('.hdf5'):
                    #print(os.path.join(m_dir, f))
                    with h5py.File(os.path.join(m_dir, f), 'r') as h5f:
                        self.data_num += h5f['{}_img'.format(self.pre)].shape[0]
                        self.data_indices[self.data_num] = os.path.join(m_dir, f)
                    count += 1
                    #if count > 1:
                    #    break
        self.weights = {}
        print('DONE SETTING UP')


    def __len__(self):
        return int(self.data_num / self.batch_size)

    def generate_class_weights(self, mu=0.5):
        from collections import Counter

        counters = {}
        weights = {}
        for k, v in self.class_counts.items():
            print(k, v)
            counters[k] = Counter()

        for i, (next_ind, path) in enumerate(self.data_indices.items()):
            with h5py.File(path, 'r') as hf5:

                for k, v in self.class_counts.items():
                    try:
                        y_train = hf5['{}_{}_label'.format(self.pre, k)]
                        unique, counts = np.unique(y_train, return_counts=True)
                        counts = dict(zip(unique, counts))
                    except KeyError:
                        counts = {self.sets[k].index('not_'+k): hf5['{}_img'.format(self.pre)].shape[0]}
                    counters[k].update(counts)

        self.weights = {}
        for k in self.sets.keys():
            total = np.sum(np.array(list(counters[k].values())))
            for k2, v2 in counters[k].items():
                print(self.sets[k][k2], v2)
            w = np.zeros((len(self.sets[k]),))
            for k2, v in counters[k].items():
                score = total/float(v)
                w[k2] = score if score <= 1000 else 1000
            self.weights[k] = torch.from_numpy(w).float()
        print('DONE SETTING UP WEIGHTS')
        return self.weights

    def __getitem__(self, index):
        start_ind = 0
        real_index = index * self.batch_size
        for i, (next_ind, v) in enumerate(self.data_indices.items()):
            path = v
            if real_index < next_ind:
                break
            start_ind = next_ind
        end = real_index + self.batch_size
        next_file = end > next_ind
        real_index = real_index - start_ind
        inputs = {}
        outputs = {}
        with h5py.File(path, 'r') as hf5:
            #if hf5['{}_hero_label'.format(self.pre)][real_index] == 0:
            #    for i in range(self.batch_size):
            #        print('index', i)
            #        for k,s in self.sets.items():
            #            print(k, s[hf5['{}_{}_label'.format(self.pre, k)][real_index+i]])
            #        print('round', path)
            #        print('time_point', hf5['{}_time_point'.format(self.pre)][real_index+i])
            #        cv2.imshow('frame_{}'.format(i), np.transpose(hf5['{}_img'.format(self.pre)][real_index+i, ...], (1, 2, 0)))
            #    cv2.waitKey()
            inputs['image']= torch.from_numpy(hf5['{}_img'.format(self.pre)][real_index:real_index+self.batch_size, ...]).float()
            inputs['round'] = hf5['{}_round'.format(self.pre)][real_index:real_index+self.batch_size]
            inputs['time_point'] = hf5['{}_time_point'.format(self.pre)][real_index:real_index+self.batch_size]
            for k in self.input_sets.keys():
                inputs[k]= torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][real_index:real_index+self.batch_size, ...]).long()

            for k in self.sets.keys():
                try:
                    outputs[k] = torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][real_index:real_index+self.batch_size]).long()
                except KeyError:
                    n = np.zeros((inputs['image'].size(0),))
                    n[:] = self.sets[k].index('not_'+ k)
                    outputs[k] = torch.from_numpy(n).long()
        if next_file:
            from_next = self.batch_size - inputs['image'].size(0)
            next_path = list(self.data_indices.values())[i+1]
            with h5py.File(next_path, 'r') as hf5:
                im = torch.from_numpy(hf5['{}_img'.format(self.pre)][0:from_next, ...]).float()
                inputs['image'] = torch.cat((inputs['image'], im), 0)
                inputs['round'] = np.concatenate((inputs['round'], hf5['{}_round'.format(self.pre)][0:from_next]), 0)
                inputs['time_point'] = np.concatenate((inputs['time_point'], hf5['{}_time_point'.format(self.pre)][0:from_next]), 0)

                for k in self.input_sets.keys():
                    x = torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][0:from_next, ...]).long()
                    inputs[k] = torch.cat((inputs[k], x), 0)

                for k in self.sets.keys():
                    try:
                        n = hf5['{}_{}_label'.format(self.pre, k)][0:from_next]
                    except KeyError:
                        n = np.zeros((from_next,))
                        n[:] = self.sets[k].index('not_'+ k)
                    outputs[k] = torch.cat((outputs[k],  torch.from_numpy(n).long()), 0)
        inputs['image'] = ((inputs['image'] / 255) - 0.5) / 0.5
        return inputs, outputs


class GameCNNHDF5Dataset(CNNHDF5Dataset):
    def __getitem__(self, index):
        start_ind = 0
        real_index = index * self.batch_size
        for i, (next_ind, v) in enumerate(self.data_indices.items()):
            path = v
            if real_index < next_ind:
                break
            start_ind = next_ind
        end = real_index + self.batch_size
        next_file = end > next_ind
        real_index = real_index - start_ind
        inputs = {}
        outputs = {}
        with h5py.File(path, 'r') as hf5:
            #if hf5['{}_hero_label'.format(self.pre)][real_index] == 0:
            #    for i in range(self.batch_size):
            #        print('index', i)
            #        for k,s in self.sets.items():
            #            print(k, s[hf5['{}_{}_label'.format(self.pre, k)][real_index+i]])
            #        print('round', path)
            #        print('time_point', hf5['{}_time_point'.format(self.pre)][real_index+i])
            #        cv2.imshow('frame_{}'.format(i), np.transpose(hf5['{}_img'.format(self.pre)][real_index+i, ...], (1, 2, 0)))
            #    cv2.waitKey()
            inputs['image']= torch.from_numpy(hf5['{}_img'.format(self.pre)][real_index:real_index+self.batch_size, ...]).float()
            inputs['vod'] = hf5['{}_vod'.format(self.pre)][real_index:real_index+self.batch_size]
            inputs['time_point'] = hf5['{}_time_point'.format(self.pre)][real_index:real_index+self.batch_size]
            for k in self.input_sets.keys():
                inputs[k]= torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][real_index:real_index+self.batch_size, ...]).long()

            for k in self.sets.keys():
                try:
                    outputs[k] = torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][real_index:real_index+self.batch_size]).long()
                except KeyError:
                    n = np.zeros((inputs['image'].size(0),))
                    n[:] = self.sets[k].index('not_'+ k)
                    outputs[k] = torch.from_numpy(n).long()
        if next_file:
            from_next = self.batch_size - inputs['image'].size(0)
            next_path = list(self.data_indices.values())[i+1]
            with h5py.File(next_path, 'r') as hf5:
                im = torch.from_numpy(hf5['{}_img'.format(self.pre)][0:from_next, ...]).float()
                inputs['image'] = torch.cat((inputs['image'], im), 0)
                inputs['vod'] = np.concatenate((inputs['vod'], hf5['{}_vod'.format(self.pre)][0:from_next]), 0)
                inputs['time_point'] = np.concatenate((inputs['time_point'], hf5['{}_time_point'.format(self.pre)][0:from_next]), 0)

                for k in self.input_sets.keys():
                    x = torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][0:from_next, ...]).long()
                    inputs[k] = torch.cat((inputs[k], x), 0)

                for k in self.sets.keys():
                    try:
                        n = hf5['{}_{}_label'.format(self.pre, k)][0:from_next]
                    except KeyError:
                        n = np.zeros((from_next,))
                        n[:] = self.sets[k].index('not_'+ k)
                    outputs[k] = torch.cat((outputs[k],  torch.from_numpy(n).long()), 0)
        inputs['image'] = ((inputs['image'] / 255) - 0.5) / 0.5
        return inputs, outputs


class KFExistsCNNHDF5Dataset(CNNHDF5Dataset):
    def __init__(self, train_dir, sets, input_sets=None, batch_size=100, pre='train', modes=None):
        if modes is None:
            modes = ['original']
        self.batch_size = batch_size
        self.pre = pre
        self.sets = sets
        if input_sets is None:
            input_sets = {}
        self.input_sets = input_sets
        self.class_counts = {}
        for k, v in sets.items():
            self.class_counts[k] = len(v)
        self.data_num = 0
        self.data_indices = {}
        count = 0
        for m in modes:
            m_dir = os.path.join(train_dir, m)
            for f in os.listdir(m_dir):
                if f.endswith('exists.hdf5'):
                    #print(os.path.join(m_dir, f))
                    with h5py.File(os.path.join(m_dir, f), 'r') as h5f:
                        self.data_num += h5f['{}_img'.format(self.pre)].shape[0]
                        self.data_indices[self.data_num] = os.path.join(m_dir, f)
                    count += 1
                    #if count > 1:
                    #    break
        self.weights = {}
        print('DONE SETTING UP')


class StatusCNNHDF5Dataset(CNNHDF5Dataset):
    def __getitem__(self, index):
        start_ind = 0
        real_index = index * self.batch_size
        for i, (next_ind, v) in enumerate(self.data_indices.items()):
            path = v
            if real_index < next_ind:
                break
            start_ind = next_ind
        end = real_index + self.batch_size
        next_file = end > next_ind
        real_index = real_index - start_ind
        inputs = {}
        outputs = {}
        with h5py.File(path, 'r') as hf5:
            inputs['image']= torch.from_numpy(hf5['{}_img'.format(self.pre)][real_index:real_index+self.batch_size, ...]).float()
            inputs['color']= torch.from_numpy(hf5['{}_color'.format(self.pre)][real_index:real_index+self.batch_size, ...]).float()
            inputs['enemy_color']= torch.from_numpy(hf5['{}_enemy_color'.format(self.pre)][real_index:real_index+self.batch_size, ...]).float()
            inputs['round'] = hf5['{}_round'.format(self.pre)][real_index:real_index+self.batch_size]
            inputs['time_point'] = hf5['{}_time_point'.format(self.pre)][real_index:real_index+self.batch_size]
            for k in self.input_sets.keys():
                inputs[k]= torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][real_index:real_index+self.batch_size, ...]).long()

            for k in self.sets.keys():
                try:
                    outputs[k] = torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][real_index:real_index+self.batch_size]).long()
                except KeyError:
                    n = np.zeros((inputs['image'].size(0),))
                    n[:] = self.sets[k].index('not_'+ k)
                    outputs[k] = torch.from_numpy(n).long()
        if next_file:
            from_next = self.batch_size - inputs['image'].size(0)
            next_path = list(self.data_indices.values())[i+1]
            with h5py.File(next_path, 'r') as hf5:
                im = torch.from_numpy(hf5['{}_img'.format(self.pre)][0:from_next, ...]).float()
                col = torch.from_numpy(hf5['{}_color'.format(self.pre)][0:from_next, ...]).float()
                ecol = torch.from_numpy(hf5['{}_enemy_color'.format(self.pre)][0:from_next, ...]).float()
                inputs['image'] = torch.cat((inputs['image'], im), 0)
                inputs['color'] = torch.cat((inputs['color'], col), 0)
                inputs['enemy_color'] = torch.cat((inputs['enemy_color'], ecol), 0)
                inputs['round'] = np.concatenate((inputs['round'], hf5['{}_round'.format(self.pre)][0:from_next]), 0)
                inputs['time_point'] = np.concatenate((inputs['time_point'], hf5['{}_time_point'.format(self.pre)][0:from_next]), 0)

                for k in self.input_sets.keys():
                    x = torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][0:from_next, ...]).long()
                    inputs[k] = torch.cat((inputs[k], x), 0)

                for k in self.sets.keys():
                    try:
                        n = hf5['{}_{}_label'.format(self.pre, k)][0:from_next]
                    except KeyError:
                        n = np.zeros((im.size(0),))
                        n[:] = self.sets[k].index('not_'+ k)
                    outputs[k] = torch.cat((outputs[k],  torch.from_numpy(n).long()), 0)
        inputs['image'] = ((inputs['image'] / 255) - 0.5) / 0.5
        #print(inputs['image'].size())
        #error
        inputs['color'] = ((inputs['color'] / 255) - 0.5) / 0.5
        inputs['enemy_color'] = ((inputs['color'] / 255) - 0.5) / 0.5
        return inputs, outputs