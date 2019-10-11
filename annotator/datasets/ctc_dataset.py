from torch.utils.data import Dataset
import torch
import lmdb
import h5py
import os
import pickle
import sys
import numpy as np


class CTCHDF5Dataset(Dataset):
    def __init__(self, train_dir, batch_size, blank_ind, pre='train', recent=False):
        self.pre = pre
        self.batch_size = batch_size
        self.blank_ind = blank_ind
        self.data_num = 0
        self.data_indices = {}
        count = 0
        for f in os.listdir(train_dir):
            if f.endswith('.hdf5'):
                if recent and int(f.replace('.hdf5', '')) < 9400:
                    continue
                with h5py.File(os.path.join(train_dir, f), 'r') as h5f:
                    self.data_num += h5f['{}_img'.format(self.pre)].shape[0]
                    self.data_indices[self.data_num] = os.path.join(train_dir, f)
                count += 1
                #if count > 2:
                #    break
        self.weights = {}
        print('DONE SETTING UP')

    def __len__(self):
        return int(self.data_num / self.batch_size)

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
            lengths = hf5["{}_label_sequence_length".format(self.pre)][real_index:real_index+self.batch_size]
            im = hf5['{}_img'.format(self.pre)][real_index:real_index+self.batch_size, ...]
            rd = hf5['{}_round'.format(self.pre)][real_index:real_index+self.batch_size, ...]
            specs = hf5['{}_spectator_mode_label'.format(self.pre)][real_index:real_index+self.batch_size, ...]
            labs = hf5["{}_label_sequence".format(self.pre)][real_index:real_index+self.batch_size, ...].astype(np.int16)
            # For removing all blank images
            inds = lengths != 1

            im = im[inds]
            rd = rd[inds]
            specs = specs[inds]
            labs = labs[inds]
            lengths = lengths[inds]

            labs[labs > self.blank_ind] = self.blank_ind
            labs = labs.reshape(labs.shape[0] * labs.shape[1])
            labs = labs[labs != self.blank_ind]
            labs += 1
            inputs['image']= torch.from_numpy(im).float()
            inputs['spectator_mode'] = torch.from_numpy(specs).long()
            inputs['round'] = torch.from_numpy(rd).long()
            outputs['the_labels'] = torch.from_numpy(labs).long()
            outputs['label_length'] = torch.from_numpy(lengths).long()
        if next_file:
            from_next = end - next_ind
            next_path = list(self.data_indices.values())[i+1]
            with h5py.File(next_path, 'r') as hf5:
                lengths = hf5["{}_label_sequence_length".format(self.pre)][0:from_next]
                im= hf5['{}_img'.format(self.pre)][0:from_next, ...]
                rd= hf5['{}_round'.format(self.pre)][0:from_next, ...]
                specs= hf5['{}_spectator_mode_label'.format(self.pre)][0:from_next, ...]
                labs = hf5["{}_label_sequence".format(self.pre)][0:from_next, ...].astype(np.int16)
                # For removing all blank images
                inds = lengths != 1

                im = im[inds]
                rd = rd[inds]
                specs = specs[inds]
                labs = labs[inds]
                lengths = lengths[inds]

                labs[labs > self.blank_ind] = self.blank_ind
                labs = labs.reshape(labs.shape[0] * labs.shape[1])
                labs = labs[labs != self.blank_ind]
                labs += 1
                inputs['image' ]= torch.cat((inputs['image'], torch.from_numpy(im).float()), 0)
                inputs['spectator_mode'] = torch.cat((inputs['spectator_mode'], torch.from_numpy(specs).long()), 0)
                inputs['round'] = torch.cat((inputs['round'], torch.from_numpy(rd).long()), 0)

                outputs['the_labels'] = torch.cat((outputs['the_labels'], torch.from_numpy(labs).long()), 0)
                outputs['label_length'] = torch.cat((outputs['label_length'], torch.from_numpy(lengths).long()), 0)

        return inputs, outputs


class LabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, label_set):
        self.label_set = label_set + ['-']  # for `-1` index

        self.dict = {}
        for i, char in enumerate(label_set):
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        '''
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
        '''
        length = []
        result = []
        for item in text:
            try:
                item = item.split(',')
                length.append(len(item))
                for char in item:
                    index = self.dict[char] + 1
                    result.append(index)
            except:
                print(item)
                raise
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ','.join([self.label_set[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.label_set[t[i] - 1])
                return ','.join(char_list)
        else:
            # batch mode

            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts