import os
import csv
import cv2
import numpy as np
import h5py
import math
import random

working_dir = r'E:\Data\Overwatch\models\replay_cnn'
os.makedirs(working_dir, exist_ok=True)

train_dir = r'E:\Data\Overwatch\training_data\replay_cnn'
log_dir = os.path.join(working_dir, 'log')
hdf5_path = os.path.join(train_dir, 'dataset.hdf5')

status_hd5_path = os.path.join(train_dir, 'dataset.hdf5')

set_files = {'replay': os.path.join(train_dir, 'replay_set.txt'),
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

class_counts = {}
for k, v in sets.items():
    class_counts[k] = len(v)

spectator_modes = load_set(os.path.join(train_dir, 'spectator_mode_set.txt'))

spectator_mode_count = len(spectator_modes)

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
        input = {}
        input['main_input'] = self.hdf5_file["{}_img".format(pre)][i_s:i_e, ...]
        # print(hero_set[self.hdf5_file["{}_hero_label".format(pre)][i_s]])
        # cv2.imshow('frame', images[0, :])
        # cv2.waitKey(0)
        input['spectator_mode_input'] = np.zeros((i_e-i_s, spectator_mode_count))
        m = sparsify(self.hdf5_file["{}_spectator_mode".format(pre)][i_s:i_e], spectator_mode_count)
        for i in range(i_e-i_s):
            input['spectator_mode_input'][i] = m[i]

        output = {}
        for k, count in class_counts.items():
            output_name = '{}_output'.format(k)
            output[output_name] = sparsify(self.hdf5_file["{}_label".format(pre)][i_s:i_e], count)
        return input, output

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
        preds = model.predict_on_batch(X)
        for output_ind, (output_key, s) in enumerate(sets.items()):
            print(preds[output_ind].shape)
            cnn_inds = preds[output_ind].argmax(axis=2)
            print(cnn_inds.shape)
            for t_ind in range(X.shape[0]):
                for j in range(X.shape[1]):
                    cnn_label = s[cnn_inds[t_ind, j]]
                    actual_label = s[y['{}_output'.format(output_key)][t_ind, j].argmax(axis=0)]
                    if cnn_label != actual_label:
                        print(output_key)
                        print(cnn_label, actual_label)
                        print(gen.hdf5_file['val_round'][i_s+t_ind], time.strftime('%M:%S', time.gmtime(gen.hdf5_file['val_time_point'][i_s+t_ind])))
                        cv2.imshow('frame', np.swapaxes(X[t_ind, j,  ...], 0, 1))
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
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, TimeDistributed, CuDNNGRU

    input_shape = (40, 150, 3)
    params = {'dim_x': 140,
              'dim_y': 300,
              'dim_z': 3,
              'batch_size': 4,
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
    final_output_weights = os.path.join(working_dir, 'mid_weights.h5')
    final_output_json = os.path.join(working_dir, 'mid_model.json')

    class_weights = {}
    for k, v in class_counts.items():
        output_name = k + '_output'
        y_train = gen.hdf5_file['train_label']
        unique, counts = np.unique(y_train, return_counts=True)
        counts = dict(zip(unique, counts))
        class_weights[k] = create_class_weight(counts)

    if not os.path.exists(final_output_json):
        current_model_path = os.path.join(working_dir, 'current_mid_model.hdf5')
        if not os.path.exists(current_model_path):
            main_input = Input(shape=input_shape, name='main_input')
            spectator_mode_input = Input(shape=(spectator_mode_count,), name='spectator_mode_input')
            x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                       activation='relu')(main_input)
            x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                       activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Conv2D(128, (3, 3), activation='relu')(x)
            x = Conv2D(128, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

            x = Dropout(0.25)(x)
            x = Flatten()(x)

            x = Dense(50, activation='relu', name='representation')(x)
            frame_output = Dropout(0.5)(x)
            x = keras.layers.concatenate([frame_output, spectator_mode_input])

            outputs = []
            for k, count in class_counts.items():
                outputs.append(Dense(count, activation='softmax', name=k + '_output')(x))
            model = Model(inputs=[main_input, spectator_mode_input], outputs=outputs)

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

    print(model.summary())
    check_val_errors(model, gen)

