import os
import csv
import cv2
import numpy as np
import h5py
import math
import random
from sklearn.utils import class_weight

working_dir = r'E:\Data\Overwatch\models\kf_slot_ctc_seq'
os.makedirs(working_dir, exist_ok=True)

train_dir = r'E:\Data\Overwatch\training_data\kf_slot_ctc_seq'
log_dir = os.path.join(working_dir, 'log')
hdf5_path = os.path.join(train_dir, 'dataset.hdf5')


def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


labels = load_set(os.path.join(train_dir, 'labels.txt'))
spectator_modes = load_set(os.path.join(train_dir, 'spectator_mode_set.txt'))

class_count = len(labels)
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
        input['the_input'] = self.hdf5_file["{}_img".format(pre)][i_s:i_e, ...]
        input['the_labels'] = self.hdf5_file["{}_label_sequence".format(pre)][i_s:i_e, ...]
        input['label_length'] = np.reshape(self.hdf5_file["{}_label_sequence_length".format(pre)][i_s:i_e], (-1, 1))
        input['input_length'] = np.zeros((i_e - i_s, 1))
        input['input_length'][:] = img_w // downsample_factor - 2

        outputs = {'ctc': np.zeros([i_e - i_s])}  # dummy data for dummy loss function

        return input, outputs

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
        X, y, _ = gen._data_generation(i_s, i_e, train=False)
        print(X['main_input'].shape)
        print('specmode shape', X['spectator_mode_input'].shape)
        print(y['labels_output'].shape)
        preds = model.predict_on_batch(X)
        print(len(preds))
        print(preds[0].shape)

        actual_inds = y['labels_output'].argmax(axis=2)
        print(actual_inds.shape)
        for t_ind in range(X['main_input'].shape[0]):
            cnn_inds = preds[t_ind].argmax(axis=1)
            predicted_labels = convert_output(cnn_inds)
            actual_labels = convert_output(actual_inds[t_ind, :])
            box = X['main_input'][t_ind, ...]
            print('spec_mode', spectator_modes[X['spectator_mode_input'][t_ind, 0, :].argmax(axis=0)])
            print('predicted', predicted_labels)
            print('actual', actual_labels)
            print(gen.hdf5_file['val_round'][i_s + t_ind],
                  time.strftime('%M:%S', time.gmtime(gen.hdf5_file['val_time_point'][i_s + t_ind])))
            cv2.imshow('frame', np.swapaxes(box, 0, 1))
            cv2.waitKey(0)


def convert_output(output):
    intervals = []
    print(output.shape)
    for i in range(output.shape[0]):
        lab = labels[output[i]]
        if not intervals or lab != intervals[-1]['label']:
            intervals.append({'begin': i, 'end': i, 'label': lab})
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
                    print(gen.hdf5_file['train_round'][i_s + t_ind],
                          time.strftime('%M:%S', time.gmtime(gen.hdf5_file['train_time_point'][i_s + t_ind])))
                    cv2.imshow('frame', X[t_ind, ...])
                    cv2.waitKey(0)


def create_class_weight(labels_dict, mu=0.5):
    total = np.sum(np.array(list(labels_dict.values())))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight


def ctc_lambda_func(args):
    ls, y_pred, input_length, label_length = args

    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(ls, y_pred, input_length, label_length)


if __name__ == '__main__':
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D, Activation, Lambda, Flatten, Dropout, Dense, Input, LSTM, \
        TimeDistributed, Bidirectional, Reshape, CuDNNGRU, CuDNNLSTM, Conv1D, MaxPooling1D
    from keras.regularizers import l1_l2
    from keras import backend as K

    input_shape = (100 ,248, 32, 3)
    params = {'dim_x': 32,
              'dim_y': 210,
              'dim_z': 3,
              'batch_size': 4,
              'shuffle': True,
              'subtract_mean': False}
    num_epochs = 40
    # Datasets

    # Generators
    gen = DataGenerator(hdf5_path, **params)
    print(gen.hdf5_file['train_img'].shape)
    training_generator = gen.generate_train()
    validation_generator = gen.generate_val()
    print('set up complete')
    # Design model
    final_output_weights = os.path.join(working_dir, 'kf_weights.h5')
    final_output_json = os.path.join(working_dir, 'kf_model.json')

    img_w = input_shape[-3]
    img_h = input_shape[-2]
    pool_size = 2
    conv_filters = 32
    kernel_size = (3, 3)
    rnn_size = 512
    time_dense_size = 128
    downsample_factor = (pool_size ** 2)
    if not os.path.exists(final_output_json):
        current_model_path = os.path.join(working_dir, 'current_kf_model.h5')
        if not os.path.exists(current_model_path):
            act = 'relu'
            input_data = Input(name='the_input', shape=input_shape, dtype='float32')
            inner = TimeDistributed(Conv2D(conv_filters, kernel_size, padding='same',
                                           activation=act,  # kernel_initializer='he_normal',
                                           name='conv1'))(input_data)
            inner = TimeDistributed(Conv2D(conv_filters, kernel_size, padding='same',
                                           activation=act,  # kernel_initializer='he_normal',
                                           name='conv2'))(inner)
            inner = TimeDistributed(MaxPooling2D(pool_size=(pool_size, pool_size), name='max1'))(inner)
            inner = TimeDistributed(Dropout(0.25))(inner)
            inner = TimeDistributed(Conv2D(conv_filters * 2, kernel_size, padding='same',
                                           activation=act,  # kernel_initializer='he_normal',
                                           name='conv3'))(inner)
            inner = TimeDistributed(Conv2D(conv_filters * 2, kernel_size, padding='same',
                                           activation=act,  # kernel_initializer='he_normal',
                                           name='conv4'))(inner)
            inner = TimeDistributed(MaxPooling2D(pool_size=(pool_size, pool_size), name='max2'))(inner)
            inner = TimeDistributed(Dropout(0.25))(inner)

            conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters * 2)
            inner = TimeDistributed(Reshape(target_shape=conv_to_rnn_dims, name='reshape'))(inner)

            # cuts down input size going into RNN:
            inner = TimeDistributed(Dense(time_dense_size, activation=act, name='dense1'))(inner)

            # Two layers of bidirectional GRUs
            # GRU seems to work as well, if not better than LSTM:
            inner = TimeDistributed(CuDNNGRU(rnn_size, return_sequences=True))(inner)
            inner = TimeDistributed(CuDNNGRU(rnn_size, return_sequences=True))(inner)
            # gru_1 = CuDNNGRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
            # gru_1b = CuDNNGRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
            #             name='gru1_b')(
            #    inner)
            # gru1_merged = add([gru_1, gru_1b])
            # gru_2 = CuDNNGRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
            # gru_2b = CuDNNGRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
            #             name='gru2_b')(
            #    gru1_merged)

            # transforms RNN output to character activations:
            inner = TimeDistributed(Dense(class_count + 1, kernel_initializer='he_normal',
                                          name='dense2'))(inner)
            y_pred = TimeDistributed(Activation('softmax', name='softmax'))(inner)
            Model(inputs=input_data, outputs=y_pred).summary()

            labels_input = Input(name='the_labels', shape=(100 ,6), dtype='float32')
            input_length = Input(name='input_length', shape=(100 ,), dtype='uint8')
            label_length = Input(name='label_length', shape=(100 ,), dtype='uint8')
            # Keras doesn't currently support loss funcs with extra parameters
            # so CTC loss is implemented in a lambda layer
            loss_out = TimeDistributed(Lambda(ctc_lambda_func, output_shape=(1,), name='ctc'))(
                [labels_input, y_pred, input_length, label_length])

            model = Model(inputs=[input_data, labels_input, input_length, label_length], outputs=loss_out)
            model.summary()
            model.compile(loss=keras.losses.categorical_crossentropy,
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
                                      # class_weight=class_weights
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
    # check_train_errors(model, gen)
