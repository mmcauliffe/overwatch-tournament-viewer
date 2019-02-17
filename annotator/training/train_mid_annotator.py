import os
import csv
import cv2
import shutil
import numpy as np
import h5py
import math
import random

from annotator.config import BOX_PARAMETERS

working_dir = r'E:\Data\Overwatch\models\mid'
os.makedirs(working_dir, exist_ok=True)

train_dir = r'E:\Data\Overwatch\training_data\mid'
log_dir = os.path.join(working_dir, 'log')
hdf5_path = os.path.join(train_dir, 'dataset.hdf5')

status_hd5_path = os.path.join(train_dir, 'dataset.hdf5')

set_files = {
    'overtime': os.path.join(train_dir, 'overtime_set.txt'),
    'point_status': os.path.join(train_dir, 'point_status_set.txt'),
}

end_set_files = {
    'attacking_color': os.path.join(train_dir, 'attacking_color_set.txt'),
    #'map': os.path.join(train_dir, 'map_set.txt'),
    #'map_mode': os.path.join(train_dir, 'map_mode_set.txt'),
    #'round_number': os.path.join(train_dir, 'round_number_set.txt'),
    #'spectator_mode': os.path.join(train_dir, 'spectator_mode_set.txt'),

}

DEBUG = False

def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


sets = {}
for k, v in set_files.items():
    sets[k] = load_set(v)
    shutil.copyfile(v, v.replace(train_dir, working_dir))

end_sets = {}
for k, v in end_set_files.items():
    end_sets[k] = load_set(v)
    shutil.copyfile(v, v.replace(train_dir, working_dir))

class_counts = {}
for k, v in sets.items():
    class_counts[k] = len(v)

end_class_counts = {}
for k, v in end_sets.items():
    end_class_counts[k] = len(v)


spectator_modes = load_set(os.path.join(train_dir, 'spectator_mode_set.txt'))

spectator_mode_count = len(spectator_modes)

maps = load_set(os.path.join(train_dir, 'map_set.txt'))

map_count = len(maps)

map_modes = load_set(os.path.join(train_dir, 'map_mode_set.txt'))

map_mode_count = len(map_modes)

def sparsify(y, n_classes):
    'Returns labels in binary NumPy array'
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])


def sparsify_2d(y, n_classes):
    'Returns labels in binary NumPy array'
    s = np.zeros((y.shape[0], y.shape[1], n_classes))
    for i in range(y.shape[0]):
        for k in range(y.shape[1]):
            s[i][k][y[i][k]] = 1
    return s


def inspect(data_gen):
    pre = 'train'
    for i in range(data_gen.data_num):
        box = data_gen.hdf5_file["train_img"][i, 0, ...]
        for k, s in sets.items():
            print(k, [s[x] for x in data_gen.hdf5_file["train_{}_label".format(k)][i]])
        print(data_gen.hdf5_file['{}_round'.format(pre)][i], data_gen.hdf5_file['{}_time_point'.format(pre)][i])
        cv2.imshow('frame', box)
        cv2.waitKey(0)


class DataGenerator(object):
    def __init__(self, dim_x=140, dim_y=300, dim_z=3, batch_size=32, shuffle=True, subtract_mean=True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_num = 0
        self.val_num = 0
        self.data_indices = {}
        self.val_indices = {}
        for f in os.listdir(train_dir):
            if f.endswith('.hdf5'):
                with h5py.File(os.path.join(train_dir, f), 'r') as h5f:
                    self.data_num += h5f['train_img'].shape[0]
                    self.val_num += h5f['val_img'].shape[0]
                    self.data_indices[self.data_num] = os.path.join(train_dir, f)
                    self.val_indices[self.val_num] = os.path.join(train_dir, f)
        self.subtract_mean = subtract_mean
        self.class_weights = {}
        self.end_class_weights = {}
        #if self.subtract_mean:
        #    self.mm = self.hdf5_file["train_mean"][0, ...]
        #    self.mm = self.mm[np.newaxis, np.newaxis, ...].astype(np.uint8)

    def generate_class_weights(self):
        from collections import Counter

        counters = {}
        for k, v in class_counts.items():
            print(k, v)
            counters[k] = Counter()
        end_counters = {}
        for k, v in end_class_counts.items():
            print(k, v)
            end_counters[k] = Counter()
        start_ind = 0
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
                start_ind = next_ind
        print(counters)
        print(end_counters)
        for k, v in class_counts.items():
            self.class_weights[k] = create_class_weight(counters[k])
        for k, v in end_class_counts.items():
            self.end_class_weights[k] = create_class_weight(end_counters[k])
        print(self.class_weights)
        print(self.end_class_weights)

    def _get_exploration_order(self, train=True):
        'Generates order of exploration'
        # Find exploration order
        if train:
            batches_list = list(range(int(math.ceil(float(self.data_num) / self.batch_size))))
        else:
            batches_list = list(range(int(math.ceil(float(self.val_num) / self.batch_size))))
        if self.shuffle:
            random.shuffle(batches_list)
        return batches_list

    def _data_generation(self, i_s, i_e, train=True, check=False):
        'Generates data of batch_size samples'  # X : (n_samples, v_size, v_size, v_size, n_channels)
        orig_i_s = i_s
        orig_i_e = i_e
        if train:
            pre = 'train'
            indices = self.data_indices
        else:
            pre = 'val'
            indices = self.val_indices

        input = {}
        output = {}
        sample_weights = {}

        start_ind = 0
        for i, (next_ind, v) in enumerate(indices.items()):
            path = v
            if i_s < next_ind:
                break
            start_ind = next_ind
        i_s -= start_ind
        i_e -= start_ind
        if orig_i_e > next_ind:
            next_file = True
        else:
            next_file = False

        with h5py.File(path, 'r') as hf5:
            #sample_weights = {}
            input['main_input'] = hf5["{}_img".format(pre)][i_s:i_e, ...]
            d = hf5["{}_spectator_mode_label".format(pre)][i_s:i_e]
            d2 = hf5["{}_map_mode_label".format(pre)][i_s:i_e]
            d3 = hf5["{}_map_label".format(pre)][i_s:i_e]
            spec_input = np.zeros((d.shape[0], 100), dtype=np.uint8)
            map_input = np.zeros((d.shape[0], 100), dtype=np.uint8)
            map_mode_input = np.zeros((d.shape[0], 100), dtype=np.uint8)
            for i in range(d.shape[0]):
                spec_input[i, :] = d[i]
                map_input[i, :] = d3[i]
                map_mode_input[i, :] = d2[i]
            input['spectator_mode_input'] = sparsify_2d(spec_input, spectator_mode_count)
            input['map_input'] = sparsify_2d(map_input, map_count)
            input['map_mode_input'] = sparsify_2d(map_mode_input, map_mode_count)
            if DEBUG or check:
                input['time_point'] = hf5["{}_time_point".format(pre)][i_s:i_e]
                input['round'] = hf5["{}_round".format(pre)][i_s:i_e]
                print(input['time_point'])
            for k, count in class_counts.items():
                output_name = '{}_output'.format(k)
                output[output_name] = sparsify_2d(hf5["{}_{}_label".format(pre, k)][i_s:i_e], count)
                #sample_weights[output_name] = np.ones((i_e-i_s,100))
                #for i in range(sample_weights[output_name].shape[0]):
                #    for j in range(sample_weights[output_name].shape[1]):
                #        sample_weights[output_name][i, j] = class_weights[k][self.hdf5_file["{}_{}_label".format(pre, k)][i+i_s, j]]

            for k, count in end_class_counts.items():
                output_name = '{}_output'.format(k)
                output[output_name] = sparsify(hf5["{}_{}_label".format(pre, k)][i_s:i_e], count)
            if DEBUG:
                for i in range(input['main_input'].shape[0]):
                    for k, s in sets.items():
                        print(k, s[hf5["{}_{}_label".format(pre, k)][i_s+i, 0]])
                    for k, s in end_sets.items():
                        print(k, s[hf5["{}_{}_label".format(pre, k)][i_s+i]])
                    print('spectator_mode', spectator_modes[hf5["{}_spectator_mode_label".format(pre)][i_s+i]])
                    print(hf5['{}_round'.format(pre)][i_s+i], hf5['{}_time_point'.format(pre)][i_s+i])
                    cv2.imshow('frame', input['main_input'][i, 0, :])
                    cv2.waitKey()

            for k, count in class_counts.items():
                output_name = k + '_output'
                data = hf5["{}_{}_label".format(pre, k)][i_s:i_e]
                sample_weights[output_name] = np.ones((i_e-i_s,))
                for i in range(data.shape[0]):
                    weight = np.mean([self.class_weights[k][data[i, j]] for j in range(data.shape[1])])
                    sample_weights[i] = weight
                output[output_name] = sparsify_2d(data, count)

            for k, count in end_class_counts.items():
                output_name = k + '_output'
                data = hf5["{}_{}_label".format(pre, k)][i_s:i_e]

                sample_weights[output_name] = np.ones((i_e-i_s,))
                for i in range(data.shape[0]):
                    sample_weights[output_name][i] = self.end_class_weights[k][data[i]]
                output[output_name] = sparsify(data, count)
        if next_file:
            from_next = orig_i_e - next_ind
            next_path = list(indices.values())[i+1]
            with h5py.File(next_path, 'r') as hf5:
                input['main_input'] = np.concatenate((input['main_input'], hf5["{}_img".format(pre)][0:from_next, ...]))
                d = hf5["{}_spectator_mode_label".format(pre)][0:from_next]
                d2 = hf5["{}_map_mode_label".format(pre)][0:from_next]
                d3 = hf5["{}_map_label".format(pre)][0:from_next]
                spec_input = np.zeros((from_next, 100), dtype=np.uint8)
                map_input = np.zeros((from_next, 100), dtype=np.uint8)
                map_mode_input = np.zeros((from_next, 100), dtype=np.uint8)
                for i in range(d.shape[0]):
                    spec_input[i, :] = d[i]
                    map_input[i, :] = d3[i]
                    map_mode_input[i, :] = d2[i]
                input['spectator_mode_input'] = np.concatenate((input['spectator_mode_input'],
                                                                sparsify_2d(spec_input, spectator_mode_count)))
                input['map_input'] = np.concatenate((input['map_input'],
                                                                sparsify_2d(map_input, map_count)))
                input['map_mode_input'] = np.concatenate((input['map_mode_input'],
                                                                sparsify_2d(map_mode_input, map_mode_count)))
                if check:
                    input['time_point'] = np.concatenate((input['time_point'], hf5["{}_time_point".format(pre)][0:from_next]))
                    input['round'] = np.concatenate((input['round'], hf5["{}_round".format(pre)][0:from_next]))
                    print(input['time_point'])

                for k, count in class_counts.items():
                    output_name = k + '_output'
                    data = hf5["{}_{}_label".format(pre, k)][0:from_next]
                    ind = 0
                    for i in range(sample_weights[output_name].shape[0] - from_next, sample_weights[output_name].shape[0]):
                        weight = np.mean([self.class_weights[k][data[ind, j]] for j in range(data.shape[1])])
                        sample_weights[i] = weight
                        ind += 1
                    output[output_name] = np.concatenate((output[output_name], sparsify_2d(hf5["{}_{}_label".format(pre, k)][0:from_next], count)))

                for k, count in end_class_counts.items():
                    output_name = k + '_output'
                    data = hf5["{}_{}_label".format(pre, k)][0:from_next]
                    ind = 0
                    for i in range(sample_weights[output_name].shape[0] - from_next, sample_weights[output_name].shape[0]):
                        sample_weights[output_name][i] = self.end_class_weights[k][data[ind]]
                        ind += 1
                    output[output_name] = np.concatenate((output[output_name], sparsify(data, count)))
        return input, output#, sample_weights

    def generate_train(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            batches_list = self._get_exploration_order()

            for n, i in enumerate(batches_list):
                i_s = i * self.batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * self.batch_size, self.data_num])  # index of the last image in this batch
                yield self._data_generation(i_s, i_e)

    def generate_val(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            batches_list = self._get_exploration_order(train=False)

            for n, i in enumerate(batches_list):
                i_s = i * self.batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * self.batch_size, self.val_num])  # index of the last image in this batch
                yield self._data_generation(i_s, i_e, train=False)


def check_val_errors(model, gen):
    import time
    # Generate order of exploration of dataset
    batches_list = gen._get_exploration_order(train=False)

    for n, i in enumerate(batches_list):
        i_s = i * gen.batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * gen.batch_size, gen.val_num])  # index of the last image in this batch
        X, y, w = gen._data_generation(i_s, i_e, train=False, check=True)
        print(X['main_input'].shape)
        preds = model.predict_on_batch(X)
        for output_ind, (output_key, s) in enumerate(list(sets.items()) + list(end_sets.items())):
            if output_key in end_sets:
                cnn_inds = preds[output_ind].argmax(axis=1)
                for t_ind in range(X['main_input'].shape[0]):
                    cnn_label = s[cnn_inds[t_ind]]
                    actual_label = s[y['{}_output'.format(output_key)][t_ind].argmax(axis=0)]
                    print(output_key, cnn_label, actual_label)

            else:
                print(preds[output_ind].shape)
                cnn_inds = preds[output_ind].argmax(axis=2)
                print(cnn_inds.shape)
                for t_ind in range(X['main_input'].shape[0]):
                    for j in range(X['main_input'].shape[1]):
                        cnn_label = s[cnn_inds[t_ind, j]]
                        actual_label = s[y['{}_output'.format(output_key)][t_ind, j].argmax(axis=0)]
                        if cnn_label != actual_label:
                            print(output_key)
                            print(cnn_label, actual_label)
                            print(X['time_point'])
                            time_point = round(X['time_point'][t_ind] - (100-j) * 0.1, 1)
                            print(X['round'][t_ind], time.strftime('%M:%S', time.gmtime(time_point)), time_point)
                            if gen.subtract_mean:
                                cv2.imshow('frame', X['main_input'][t_ind, j, ...] + gen.mm[0,  0, ...])
                            else:
                                cv2.imshow('frame', X['main_input'][t_ind, j, ...])
                            cv2.waitKey(0)


def check_train_errors(model, gen):
    import time
    # Generate order of exploration of dataset
    batches_list = gen._get_exploration_order(train=False)

    for n, i in enumerate(batches_list):
        i_s = i * gen.batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * gen.batch_size, gen.data_num])  # index of the last image in this batch
        X, y = gen._data_generation(i_s, i_e, train=True)
        print(X.shape)
        preds = model.predict_on_batch(X)
        for output_ind, (output_key, s) in enumerate(sets.items()):
            if output_key != 'second_hero':
                continue
            output_ind = 0
            cnn_inds = preds.argmax(axis=1)
            for t_ind in range(X.shape[0]):
                cnn_label = s[cnn_inds[t_ind]]
                actual_label = s[y['{}_output'.format(output_key)][t_ind].argmax(axis=0)]
                if cnn_label != actual_label:
                    print(output_key)
                    print(cnn_label, actual_label)
                    print(gen.hdf5_file['train_round'][i_s + t_ind],
                          time.strftime('%M:%S', time.gmtime(gen.hdf5_file['train_time_point'][i_s + t_ind])))
                    cv2.imshow('frame_subtracted', X[t_ind, ...])
                    cv2.imshow('frame', X[t_ind, ...] + gen.mm[0,  0, ...])
                    cv2.imshow('mean', gen.mm[0,  0, ...])
                    cv2.waitKey(0)


def create_class_weight(labels_dict, mu=5):
    total = np.sum(np.array(list(labels_dict.values())))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

def transfer_label_data():
    import shutil

    for k, v in set_files.items():
        shutil.copyfile(v, v.replace(train_dir, working_dir))
    for k, v in end_set_files.items():
        shutil.copyfile(v, v.replace(train_dir, working_dir))

if __name__ == '__main__':
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, TimeDistributed, CuDNNGRU

    input_shape = (
    100, int(BOX_PARAMETERS['O']['MID']['HEIGHT'] * 0.5), int(BOX_PARAMETERS['O']['MID']['WIDTH'] * 0.5), 3)
    params = {'dim_x': 140,
              'dim_y': 300,
              'dim_z': 3,
              'batch_size': 4,
              'shuffle': True,
              'subtract_mean': False}
    num_epochs = 40
    # Datasets

    # Generators
    gen = DataGenerator(**params)
    gen.generate_class_weights()
    # inspect(gen)
    training_generator = gen.generate_train()
    if DEBUG:
        for t in training_generator:
            pass
    validation_generator = gen.generate_val()
    print('set up complete')
    # Design model
    final_output_weights = os.path.join(working_dir, 'mid_weights.h5')
    final_output_json = os.path.join(working_dir, 'mid_model.json')

    if not os.path.exists(final_output_json):
        current_model_path = os.path.join(working_dir, 'current_mid_model.hdf5')
        if not os.path.exists(current_model_path):
            main_input = Input(shape=input_shape, name='main_input')
            x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                       activation='relu'))(main_input)
            x = TimeDistributed(Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
                                       activation='relu'))(x)
            x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
            x = TimeDistributed(Dropout(0.25))(x)
            x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
            x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
            x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
            x = TimeDistributed(Dropout(0.25))(x)
            x = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(x)
            x = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(x)
            x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)

            x = TimeDistributed(Dropout(0.25))(x)
            x = TimeDistributed(Flatten())(x)

            x = TimeDistributed(Dense(100, activation='relu', name='representation'))(x)
            x = TimeDistributed(Dropout(0.5))(x)
            spectator_mode_input = Input(shape=(100, spectator_mode_count,), name='spectator_mode_input')
            #map_input = Input(shape=(100, map_count,), name='map_input')
            map_mode_input = Input(shape=(100, map_mode_count,), name='map_mode_input')
            x = keras.layers.concatenate([x, spectator_mode_input, map_mode_input])
            x = CuDNNGRU(128, return_sequences=True)(x)
            x = CuDNNGRU(128, return_sequences=True)(x)
            seq_x = CuDNNGRU(128)(x)

            outputs = []
            for k, count in class_counts.items():
                outputs.append(TimeDistributed(Dense(count, activation='softmax'), name=k + '_output')(x))
            for k, count in end_class_counts.items():
                outputs.append(Dense(count, activation='softmax', name=k + '_output')(seq_x))
            model = Model(inputs=[main_input, spectator_mode_input, map_mode_input], outputs=outputs)

            model.summary()
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
            print('model compiled')
        else:
            model = keras.models.load_model(current_model_path)
        # Train model on dataset
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=current_model_path, verbose=1, save_best_only=True)
        early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=0,
                                                      mode='auto')
        tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
        history = model.fit_generator(generator=training_generator,
                                      epochs=num_epochs,
                                      steps_per_epoch=gen.data_num // params['batch_size'],
                                      # steps_per_epoch=100,
                                      validation_data=validation_generator,
                                      validation_steps=gen.val_num // params['batch_size'],
                                      # validation_steps=100
                                      class_weight=gen.class_weights,
                                      callbacks=[checkpointer, early_stopper, tensorboard]
                                      )
        model.save_weights(final_output_weights)
        model_json = model.to_json()
        with open(final_output_json, "w") as json_file:
            json_file.write(model_json)
    else:
        with open(final_output_json, 'r') as f:
            loaded_model_json = f.read()
        model = keras.models.model_from_json(loaded_model_json)
        model.load_weights(final_output_weights)
    transfer_label_data()

    print(model.summary())
    check_val_errors(model, gen)
