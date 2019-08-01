from torch.utils.data import Dataset
import torch
import lmdb
import os
import h5py
import pickle
import sys
from collections import Counter
import numpy as np


class CNNDataset(Dataset):
    def __init__(self, root, sets, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)
        self.sets = sets
        self.class_counts = {}
        for k, v in sets.items():
            self.class_counts[k] = len(v)
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform
        self.weights = {}

    def __len__(self):
        return self.nSamples

    def generate_class_weights(self, train_directory, mu=0.5):
        import time
        import json
        import os
        counters = {}
        print(len(self))
        begin = time.time()
        count_paths = {}
        do_count = False
        for k in self.class_counts.keys():
            count_paths[k] = os.path.join(train_directory, '{}_counts.json'.format(k))
            if not os.path.exists(count_paths[k]):
                do_count = True
        if do_count:
            for k, v in self.class_counts.items():
                print(k, v)
                counters[k] = Counter()
            for index in range(len(self)):
                with self.env.begin(write=False) as txn:
                    #img_key = 'image-%09d' % index
                    #img = pickle.loads(txn.get(img_key.encode('utf8')))
                    for k, s in self.sets.items():
                        key = '%s-%09d' % (k, index)
                        value = txn.get(key.encode('utf8')).decode('utf8')
                        counters[k][value] += 1
                if index % 1000 == 0:
                    print('{}/{} Got in {} seconds'.format(index, len(self), time.time()-begin))
                    begin = time.time()
            for k, c in counters.items():
                with open(os.path.join(train_directory, '{}_counts.json'.format(k)), 'w', encoding='utf8') as f:
                    json.dump(c, f, indent=4)
        else:
            for k,v in count_paths.items():
                with open(os.path.join(train_directory, '{}_counts.json'.format(k)), 'r', encoding='utf8') as f:
                    counters[k] = json.load(f)
        for k, v in counters.items():
            counters[k] = {self.sets[k].index(k2): v2 for k2, v2 in v.items()}
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
        assert index <= len(self), 'index range error'
        #index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            img = pickle.loads(txn.get(img_key.encode('utf8')))

            if self.transform is not None:
                img = self.transform(img)
            inputs = {'image': torch.from_numpy(img).long()}
            outputs = {}
            for k, s in self.sets.items():
                key = '%s-%09d' % (k, index)
                value = txn.get(key.encode('utf8')).decode('utf8')
                outputs[k] = s.index(value)
        return inputs, outputs


class BatchedCNNDataset(CNNDataset):
    def __init__(self, root, sets, batch_size, transform=None, target_transform=None):
        if not batch_size:
            raise Exception('Specify a batch size')
        super(BatchedCNNDataset, self).__init__(root, sets, transform=transform, target_transform=target_transform)
        self.batch_size = batch_size

    def __len__(self):
        return int(self.nSamples / self.batch_size)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        #index += 1
        real_index = index * self.batch_size
        inputs = {'image': []}
        outputs = {}
        for k in self.sets.keys():
            outputs[k] = []

        with self.env.begin(write=False) as txn:
            for i in range(self.batch_size):
                index = real_index + i
                img_key = 'image-%09d' % index
                img = pickle.loads(txn.get(img_key.encode('utf8')))

                if self.transform is not None:
                    img = self.transform(img)
                inputs['image'].append(torch.from_numpy(img).float())
                for k, s in self.sets.items():
                    key = '%s-%09d' % (k, index)
                    value = txn.get(key.encode('utf8')).decode('utf8')
                    outputs[k].append(s.index(value))
        inputs['image'] = torch.stack(inputs['image'])
        for k, v in outputs.items():
            outputs[k] = torch.LongTensor(v)
        return inputs, outputs


class CNNHDF5Dataset(Dataset):
    def __init__(self, train_dir, sets, batch_size, pre='train', recent=False):
        self.batch_size = batch_size
        self.pre = pre
        self.sets = sets
        self.class_counts = {}
        for k, v in sets.items():
            self.class_counts[k] = len(v)
        self.data_num = 0
        self.data_indices = {}
        count = 0
        for f in os.listdir(train_dir):
            if f.endswith('.hdf5'):
                if recent and int(f.replace('.hdf5', '')) < 9359:
                    continue
                with h5py.File(os.path.join(train_dir, f), 'r') as h5f:
                    self.data_num += h5f['{}_img'.format(self.pre)].shape[0]
                    self.data_indices[self.data_num] = os.path.join(train_dir, f)
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

            inputs['image']= torch.from_numpy(hf5['{}_img'.format(self.pre)][real_index:real_index+self.batch_size, ...]).float()

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
                for k in self.sets.keys():
                    try:
                        n = hf5['{}_{}_label'.format(self.pre, k)][0:from_next]
                    except KeyError:
                        n = np.zeros((from_next,))
                        n[:] = self.sets[k].index('not_'+ k)
                    outputs[k] = torch.cat((outputs[k],  torch.from_numpy(n).long()), 0)

        return inputs, outputs
