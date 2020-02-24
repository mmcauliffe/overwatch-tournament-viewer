from torch.utils.data import Dataset
import torch
import os
import h5py
import numpy as np

def test_dataset(train_dir):
    error_files = []
    for f in os.listdir(train_dir):
        if f.endswith('.hdf5'):
            print(f)
            try:
                with h5py.File(os.path.join(train_dir, f), 'r') as hf5:
                    print('train shape', hf5['train_img'])
                    print('val shape', hf5['val_img'])
            except RuntimeError:
                print('Errored')
                error_files.append(f)
    if error_files:
        print(error_files)
        raise Exception


class CNNHDF5Dataset(Dataset):
    def __init__(self, train_dir, sets, input_sets=None, batch_size=100, pre='train', recent=False, num_files=None):
        self.batch_size = batch_size
        self.pre = pre
        self.sets = sets
        if input_sets is None:
            input_sets = {}
        self.input_sets = input_sets
        self.class_counts = {}
        for k, v in sets.items():
            self.class_counts[k] = len(v)
        count = 0
        self.images = None
        self.outputs = {}
        self.inputs = {}
        for f in os.listdir(train_dir):
            if f.endswith('.hdf5'):
                if recent and int(f.replace('.hdf5', '')) < 9359:
                    continue
                print(os.path.join(train_dir, f))
                try:
                    with h5py.File(os.path.join(train_dir, f), 'r') as hf5:
                        x = torch.from_numpy(hf5['{}_img'.format(self.pre)][:]).float()
                        if self.images is None:
                            self.images = x
                        else:
                            self.images = torch.cat((self.images, x), 0)

                        for k in self.input_sets.keys():
                            x = torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][:]).long()
                            if k not in self.inputs:
                                self.inputs[k] = x
                            else:
                                self.inputs[k] = torch.cat((self.inputs[k], x), 0)

                        for k in self.sets.keys():
                            x = torch.from_numpy(hf5['{}_{}_label'.format(self.pre, k)][:]).long()
                            if k not in self.outputs:
                                self.outputs[k] = x
                            else:
                                self.outputs[k] = torch.cat((self.outputs[k], x), 0)
                except RuntimeError:
                    pass
                count += 1
                if num_files and count >= num_files:
                    break
        print('images', self.images.size())
        for k, v in self.inputs.items():
            print('k', v.size())
        for k, v in self.outputs.items():
            print('k', v.size())
        self.weights = {}
        print('DONE SETTING UP')

    def __len__(self):
        return int(self.images.shape[0] / self.batch_size)

    def generate_class_weights(self, mu=0.5):
        from collections import Counter

        counters = {}
        weights = {}
        for k, v in self.class_counts.items():
            print(k, v)
            counters[k] = Counter()

        for k, v in self.class_counts.items():
            unique, counts = np.unique(self.outputs[k], return_counts=True)
            counts = dict(zip(unique, counts))
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
        inputs = {'image': self.images[index, ...]}
        for k in self.input_sets.keys():
            inputs[k]= self.inputs[k][index, ...]
        outputs = {}
        for k in self.sets.keys():
            outputs[k]= self.outputs[k][index, ...]
        return inputs, outputs
