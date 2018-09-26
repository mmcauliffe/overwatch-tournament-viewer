import os
import csv
import cv2
import h5py
import math
import itertools
import random
import editdistance
import numpy as np
import datetime
from scipy import ndimage
import pylab
from sklearn.utils import class_weight
import keras

working_dir = r'E:\Data\Overwatch\models\kill_feed_ctc'
os.makedirs(working_dir, exist_ok=True)

train_dir = r'E:\Data\Overwatch\training_data\kill_feed_ctc'
log_dir = os.path.join(working_dir, 'log')
hdf5_path = os.path.join(train_dir, 'dataset.hdf5')


def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


labels = load_set(os.path.join(train_dir, 'labels_set.txt'))
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
        debug = False
        input['the_input'] = self.hdf5_file["{}_img".format(pre)][i_s:i_e, ...]
        input['spectator_mode_input'] = np.zeros((i_e-i_s, 62, spectator_mode_count))
        m = sparsify(self.hdf5_file["{}_spectator_mode".format(pre)][i_s:i_e], spectator_mode_count)
        for i in range(62):
            input['spectator_mode_input'][:,i, :] = m
        input['the_labels'] = self.hdf5_file["{}_label_sequence".format(pre)][i_s:i_e, ...]
        input['label_length'] = np.reshape(self.hdf5_file["{}_label_sequence_length".format(pre)][i_s:i_e], (-1, 1))
        input['input_length'] = np.zeros((i_e - i_s, 1))
        input['input_length'][:] = img_w // downsample_factor - 2
        #input['round'] = self.hdf5_file['{}_round'.format(pre)][i_s:i_e]
        #input['time_point'] = self.hdf5_file['{}_time_point'.format(pre)][i_s:i_e]
        if debug:
            for i in range(input['the_input'].shape[0]):
                print(input['spectator_mode_input'][i])
                print(input['the_labels'][i])
                print([labels[x] for x in input['the_labels'][i] if x < len(labels)])
                print(input['label_length'][i])
                print(input['input_length'][i])
                cv2.imshow('frame', np.swapaxes(input['the_input'][i], 0, 1))
                cv2.waitKey(0)

        outputs = {'ctc': np.zeros([i_e - i_s])}  # dummy data for dummy loss function
        sample_weights = np.ones([i_e - i_s])
        for i in range(i_e - i_s):
            weight = [label_weights[x] for x in input['the_labels'][i] if x != len(labels)]
            if not weight:
                weight = 1
            else:
                weight = np.mean(weight)
            sample_weights[i] = weight
        return input, outputs, sample_weights

    def generate_train(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            batches_list = self._get_exploration_order()

            for n, i in enumerate(batches_list):
                i_s = i * self.batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * self.batch_size, self.data_num])  # index of the last image in this batch
                X, y, weights = self._data_generation(i_s, i_e)
                yield X, y, weights

    def generate_val(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            batches_list = self._get_exploration_order(train=False)

            for n, i in enumerate(batches_list):
                i_s = i * self.batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * self.batch_size, self.val_num])  # index of the last image in this batch
                X, y,_ = self._data_generation(i_s, i_e, train=False)
                yield X, y


def check_val_errors(model, gen):
    import time
    # Generate order of exploration of dataset
    batches_list = gen._get_exploration_order(train=False)

    for n, i in enumerate(batches_list):
        i_s = i * gen.batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * gen.batch_size, gen.val_num])  # index of the last image in this batch
        X, y, _ = gen._data_generation(i_s, i_e, train=False)
        res = decode_batch(model.predict_on_batch, X['the_input'], X['spectator_mode_input'])
        print(res)

        for i in range(len(res)):
            print(X['round'][i], X['time_point'][i])
            print('actual', labels_to_text(X['the_labels'][i]))
            print('predicted', res[i])
            cv2.imshow('frame', np.swapaxes(X['the_input'][i], 0, 1))
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

def labels_to_text(ls):
    ret = []
    for c in ls:
        if c >= len(labels):
            continue
        ret.append(labels[c])
    return ret

def decode_batch(model, word_batch, spec_batch):
    out = model.predict_on_batch([word_batch, spec_batch])
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


class VizCallback(keras.callbacks.Callback):
    def __init__(self, run_name, test_func, text_img_gen, num_display_words=18):
        self.test_func = test_func
        self.output_dir = os.path.join(
            working_dir, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc], word_batch['spectator_mode_input'][0:num_proc])
            for j in range(num_proc):
                ref = [labels[x] for x in word_batch['the_labels'][j] if x < len(labels)]
                edit_dist = editdistance.eval(decoded_res[j], ref)
                mean_ed += float(edit_dist)
                n = len(ref)
                if n == 0:
                    n = 1
                mean_norm_ed += float(edit_dist) / n
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words], word_batch['spectator_mode_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T)
            pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % ('|'.join([labels[x] for x in word_batch['the_labels'][i] if x < len(labels)]), '|'.join(res[i])))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        pylab.close()

if __name__ == '__main__':
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D, Activation, Lambda, Flatten, Dropout, Dense, Input, LSTM, \
        TimeDistributed, Bidirectional, Reshape, CuDNNGRU, CuDNNLSTM, Conv1D, MaxPooling1D
    from keras.regularizers import l1_l2
    from keras import backend as K

    input_shape = (248, 32, 3)
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
    print(gen.hdf5_file['train_img'].shape)
    training_generator = gen.generate_train()
    validation_generator = gen.generate_val()
    print('set up complete')
    # Design model
    final_output_weights = os.path.join(working_dir, 'kf_weights.h5')
    final_output_json = os.path.join(working_dir, 'kf_model.json')

    y_train = gen.hdf5_file['train_label_sequence']
    unique, counts = np.unique(y_train, return_counts=True)
    counts = dict(zip(unique, counts))
    del counts[len(labels)]
    print(counts)
    print({k: v / sum(counts.values()) for k,v in counts.items()})
    label_weights = create_class_weight(counts)
    print(label_weights)
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
            inner = Conv2D(conv_filters, kernel_size, padding='same',
                                           activation=act,  # kernel_initializer='he_normal',
                                           name='conv1')(input_data)
            inner = Conv2D(conv_filters, kernel_size, padding='same',
                                           activation=act,  # kernel_initializer='he_normal',
                                           name='conv2')(inner)
            inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
            inner = Dropout(0.25)(inner)
            inner = Conv2D(conv_filters * 2, kernel_size, padding='same',
                                           activation=act,  # kernel_initializer='he_normal',
                                           name='conv3')(inner)
            inner = Conv2D(conv_filters * 2, kernel_size, padding='same',
                                           activation=act,  # kernel_initializer='he_normal',
                                           name='conv4')(inner)
            inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
            inner = Dropout(0.25)(inner)

            conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters * 2)
            inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)
            print(conv_to_rnn_dims)
            # cuts down input size going into RNN:
            inner = Dense(time_dense_size, activation=act, name='dense1')(inner)
            spectator_mode_input = Input(name='spectator_mode_input', shape=(62, spectator_mode_count))
            inner = keras.layers.concatenate([inner, spectator_mode_input])
            # Two layers of bidirectional GRUs
            # GRU seems to work as well, if not better than LSTM:
            inner = CuDNNGRU(rnn_size, return_sequences=True)(inner)
            inner = CuDNNGRU(rnn_size, return_sequences=True)(inner)
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
            inner = Dense(class_count + 1, kernel_initializer='he_normal',
                                          name='dense2')(inner)
            y_pred = Activation('softmax', name='softmax')(inner)
            Model(inputs=[input_data, spectator_mode_input], outputs=y_pred).summary()

            labels_input = Input(name='the_labels', shape=[12], dtype='float32')
            input_length = Input(name='input_length', shape=[1], dtype='uint8')
            label_length = Input(name='label_length', shape=[1], dtype='uint8')
            # Keras doesn't currently support loss funcs with extra parameters
            # so CTC loss is implemented in a lambda layer
            loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
                [labels_input, y_pred, input_length, label_length])

            model = Model(inputs=[input_data, spectator_mode_input, labels_input, input_length, label_length], outputs=loss_out)
            model.summary()
            model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        else:
            model = keras.models.load_model(current_model_path)
        print('model compiled')
        # Train model on dataset
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=current_model_path, verbose=1, save_best_only=True)
        early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=0,
                                                      mode='auto')
        tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)

        embedding_model = keras.models.Model(inputs=[model.input[0],model.input[1]],
                                             outputs=[model.get_layer('softmax').output])
        test_func = embedding_model.predict_on_batch

        run_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        viz_cb = VizCallback(run_name, test_func, validation_generator)
        history = model.fit_generator(generator=training_generator,
                                      epochs=num_epochs,
                                      steps_per_epoch=gen.data_num // params['batch_size'],
                                      # steps_per_epoch=100,
                                      validation_data=validation_generator,
                                      validation_steps=gen.val_num // params['batch_size'],
                                      # validation_steps=100
                                      callbacks=[viz_cb, checkpointer, early_stopper, tensorboard],
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

        embedding_model = keras.models.Model(inputs=[model.input[0],model.input[1]],
                                             outputs=[model.get_layer('softmax').output])
        test_func = embedding_model.predict_on_batch

    print(model.summary())
    check_val_errors(embedding_model, gen)
    # check_train_errors(model, gen)
