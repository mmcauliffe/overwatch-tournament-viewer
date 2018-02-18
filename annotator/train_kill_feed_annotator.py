import os
import csv
import cv2
import numpy as np
import h5py
import math
import random
from sklearn.utils import class_weight

working_dir = r'E:\Data\Overwatch\models\kf_cnn'
os.makedirs(working_dir, exist_ok=True)

train_dir = r'C:\Users\micha\Documents\Data\kf_cnn'
train_dir = r'E:\Data\Overwatch\training_data\kf_cnn'
log_dir = os.path.join(working_dir, 'log')
hdf5_path = os.path.join(train_dir, 'dataset.hdf5')

hero_set_file = os.path.join(train_dir, 'hero_set.txt')
color_set_file = os.path.join(train_dir, 'color_set.txt')
ability_set_file = os.path.join(train_dir, 'ability_set.txt')
headshot_set_file = os.path.join(train_dir, 'headshot_set.txt')


set_files = {
    #'first_hero': os.path.join(train_dir, 'hero_set.txt'),
    #'first_color': os.path.join(train_dir, 'color_set.txt'),
    #'headshot': os.path.join(train_dir, 'headshot_set.txt'),
    #'ability': os.path.join(train_dir, 'ability_set.txt'),
    'second_hero': os.path.join(train_dir, 'hero_set.txt'),
    'second_color': os.path.join(train_dir, 'color_set.txt'),

}

debug = False
ability_to_find = ''
hero_to_find = ''

def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts

#sets = {}
#for k, v in set_files.items():
#    sets[k] = load_set(v)

#class_counts = {}
#for k, v in sets.items():
#    class_counts[k] = len(v)

labels = load_set(os.path.join(train_dir, 'labels.txt'))

class_count = len(labels)

def sparsify(y, n_classes):
    'Returns labels in binary NumPy array'
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])

def sparsify(y, n_classes):
    'Returns labels in binary NumPy array'
    s = np.zeros((y.shape[0], y.shape[1], n_classes))
    for i in range(y.shape[0]):
        for k in range(y.shape[1]):
            s[i][k][y[i][k]] = 1
    return s

class DataGenerator(object):
    def __init__(self, hdf5_path, dim_x=140, dim_y=300, dim_z=3, batch_size=32, shuffle=True, subtract_mean=True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.data_num = self.hdf5_file["train_img"].shape[0]
        self.val_num = self.hdf5_file["val_img"].shape[0]
        self.subtract_mean = subtract_mean
        if self.subtract_mean:
            self.mm = self.hdf5_file["train_mean"][0, ...]
            self.mm = self.mm[np.newaxis, ...].astype(np.uint8)

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

    def _data_generation(self, i_s, i_e, train=True):
        'Generates data of batch_size samples'  # X : (n_samples, v_size, v_size, v_size, n_channels)
        if train:
            pre = 'train'
        else:
            pre = 'val'
        #s = self.hdf5_file["{}_img".format(pre)].shape
        #images = np.zeros((i_e-i_s,s[2], s[2], s[3]))
        #beg = int((s[2] - s[1]) /2)
        #end = beg + s[1]
        #images[:, beg:end, :, :] = self.hdf5_file["{}_img".format(pre)][i_s:i_e, :, :, :]
        images = self.hdf5_file["{}_img".format(pre)][i_s:i_e, ...]
        if self.subtract_mean:
            images -= self.mm
        #print(images.shape)
        #for i in range(images.shape[0]):
        #    for k, s in sets.items():
        #        print(k, s[self.hdf5_file["{}_{}_label".format(pre, k)][i_s+i, 0]])
        #    print(self.hdf5_file['{}_round'.format(pre)][i_s+i], self.hdf5_file['{}_time_point'.format(pre)][i_s+i])
        #    cv2.imshow('frame', images[i, 0, :])
        #    cv2.waitKey(0)
        output = {}
        output['output'] = sparsify(self.hdf5_file["{}_label".format(pre)][i_s:i_e], class_count)

        if debug:
            for i in range(i_s, i_e):
                for k, s in sets.items():
                    print(k, s[self.hdf5_file["{}_{}_label".format(pre, k)][i]])
                cv2.imshow('frame', self.hdf5_file["{}_img".format(pre)][i, :, :, :])
                cv2.waitKey(0)
            if ability_to_find:
                abilities = [sets['ability'][x] for x in self.hdf5_file["{}_{}_label".format(pre, 'ability')][i_s:i_e] ]
                try:
                    ind = abilities.index(ability_to_find)
                    cv2.imshow('frame', images[ind, ...])
                    cv2.waitKey(0)

                except ValueError:
                    pass
            if hero_to_find:
                first_heroes = [sets['first_hero'][x] for x in self.hdf5_file["{}_{}_label".format(pre, 'first_hero')][i_s:i_e] ]
                try:
                    ind = first_heroes.index(hero_to_find)
                    cv2.imshow('frame', images[ind, ...])
                    cv2.waitKey(0)

                except ValueError:
                    pass
                second_heroes = [sets['first_hero'][x] for x in self.hdf5_file["{}_{}_label".format(pre, 'second_hero')][i_s:i_e] ]
                try:
                    ind = second_heroes.index(hero_to_find)
                    cv2.imshow('frame', images[ind, ...])
                    cv2.waitKey(0)

                except ValueError:
                    pass
        return images, output

    def generate_train(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            batches_list = self._get_exploration_order()

            for n, i in enumerate(batches_list):
                i_s = i * self.batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * self.batch_size, self.data_num])  # index of the last image in this batch
                X, y = self._data_generation(i_s, i_e)
                yield X, y

    def generate_val(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            batches_list = self._get_exploration_order(train=False)

            for n, i in enumerate(batches_list):
                i_s = i * self.batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * self.batch_size, self.val_num])  # index of the last image in this batch
                X, y = self._data_generation(i_s, i_e, train=False)
                yield X, y


def check_val_errors(model, gen):
    import time
    # Generate order of exploration of dataset
    batches_list = gen._get_exploration_order(train=False)

    for n, i in enumerate(batches_list):
        i_s = i * gen.batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * gen.batch_size, gen.val_num])  # index of the last image in this batch
        X, y = gen._data_generation(i_s, i_e, train=False)
        print(X.shape)
        print(y['output'].shape)
        preds = model.predict_on_batch(X)
        print(len(preds))
        print(preds[0].shape)

        actual_inds = y['output'].argmax(axis=2)
        print(actual_inds.shape)
        for t_ind in range(X.shape[0]):
            cnn_inds = preds[t_ind].argmax(axis=1)
            predicted_labels = convert_output(cnn_inds)
            actual_labels = convert_output(actual_inds[t_ind, :])
            box = X[t_ind, ...]
            print('predicted', predicted_labels)
            print('actual', actual_labels)
            print(gen.hdf5_file['val_round'][i_s+t_ind], time.strftime('%M:%S', time.gmtime(gen.hdf5_file['val_time_point'][i_s+t_ind])))
            cv2.imshow('frame', np.swapaxes(box, 0, 1))
            cv2.waitKey(0)

def convert_output(output):
    intervals = []
    print(output.shape)
    for i in range(output.shape[0]):
        lab = labels[output[i]]
        if not intervals or lab != intervals[-1]['label']:
            intervals.append({'begin':i, 'end': i, 'label': lab})
        else:
            intervals[-1]['end'] = i
    return intervals

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
                    print(gen.hdf5_file['train_round'][i_s+t_ind], time.strftime('%M:%S', time.gmtime(gen.hdf5_file['train_time_point'][i_s+t_ind])))
                    cv2.imshow('frame', X[t_ind, ...])
                    cv2.waitKey(0)

def create_class_weight(labels_dict,mu=0.5):
    total = np.sum(np.array(list(labels_dict.values())))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

if __name__ == '__main__':
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, LSTM, TimeDistributed, Bidirectional, CuDNNGRU, CuDNNLSTM, Conv1D, MaxPooling1D

    input_shape = (250, 26, 3)
    params = {'dim_x': 32,
              'dim_y': 210,
              'dim_z': 3,
              'batch_size': 128,
              'shuffle': True,
              'subtract_mean': False}
    num_epochs = 40
    # Datasets

    # Generators
    gen = DataGenerator(hdf5_path, **params)
    training_generator = gen.generate_train()
    validation_generator = gen.generate_val()
    print('set up complete')
    # Design model
    final_output_weights = os.path.join(working_dir, 'kf_weights.h5')
    final_output_json = os.path.join(working_dir, 'kf_model.json')


    if not os.path.exists(final_output_json):
        current_model_path = os.path.join(working_dir, 'current_kf_model.h5')
        if not os.path.exists(current_model_path):
            main_input = Input(shape=input_shape, name='main_input')
            x = TimeDistributed(Conv1D(32, kernel_size=3,
                       activation='relu'))(main_input)
            x = TimeDistributed(Conv1D(32, kernel_size=3,
                       activation='relu'))(x)
            x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
            x = TimeDistributed(Dropout(0.25))(x)

            x = TimeDistributed(Conv1D(64, 3, activation='relu'))(x)
            x = TimeDistributed(Conv1D(64, 3, activation='relu'))(x)
            x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
            x = TimeDistributed(Dropout(0.25))(x)

            x = TimeDistributed(Flatten())(x)

            x = TimeDistributed(Dropout(0.5))(x)
            x = CuDNNGRU(64, return_sequences=True)(x)
            x = CuDNNGRU(64, return_sequences=True)(x)
            x = CuDNNGRU(64, return_sequences=True)(x)


            outputs = []
            losses = {}
            output_name = 'output'
            outputs.append(TimeDistributed(Dense(class_count, activation='softmax'), name=output_name)(x))
            losses[output_name] = keras.losses.categorical_crossentropy


            model = Model(inputs=[main_input], outputs=outputs)
            model.summary()
            model.compile(loss=losses,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        else:
            model = keras.models.load_model(current_model_path)
        print('model compiled')
        # Train model on dataset
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=current_model_path, verbose=1, save_best_only=True)
        early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0,
                                                      mode='auto')
        tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
        history = model.fit_generator(generator=training_generator,
                            epochs=num_epochs,
                            steps_per_epoch=gen.data_num // params['batch_size'],
                            # steps_per_epoch=100,
                            validation_data=validation_generator,
                            validation_steps=gen.val_num // params['batch_size'],
                            # validation_steps=100
                            callbacks=[checkpointer, early_stopper, tensorboard],
                                      #class_weight=class_weights
                            )
        model.save_weights(final_output_weights)
        model_json = model.to_json()
        with open(final_output_json, "w") as json_file:
            json_file.write(model_json)
        # list all data in history
    else:
        with open(final_output_json, 'r') as f:
            loaded_model_json = f.read()
        model = keras.models.model_from_json(loaded_model_json)
        model.load_weights(final_output_weights)

    print(model.summary())
    check_val_errors(model, gen)
    #check_train_errors(model, gen)