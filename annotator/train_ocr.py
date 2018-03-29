import os
import csv
import cv2
import numpy as np
import h5py
import math
import random
import os
import itertools
import codecs
import re
import datetime
import editdistance
import numpy as np
from scipy import ndimage
import pylab
from sklearn.utils import class_weight
import keras

working_dir = r'E:\Data\Overwatch\models\player_ocr_ctc'
os.makedirs(working_dir, exist_ok=True)

train_dir = r'E:\Data\Overwatch\training_data\player_status_cnn'
log_dir = os.path.join(working_dir, 'log')
hdf5_path = os.path.join(train_dir, 'ocr_dataset.hdf5')


def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


characters = load_set(os.path.join(train_dir, 'characters.txt'))

class_count = len(characters)

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
        input['the_input'] = np.expand_dims(self.hdf5_file["{}_img".format(pre)][i_s:i_e, ...], -1)

        input['the_labels'] = self.hdf5_file["{}_label_sequence".format(pre)][i_s:i_e, ...]
        input['label_length'] = np.reshape(self.hdf5_file["{}_label_sequence_length".format(pre)][i_s:i_e], (-1, 1))

        for i in range(input['the_labels'].shape[0]):
            if input['label_length'][i] == 0:
                input['label_length'][i] = 1
                input['the_labels'][i] = class_count
        input['input_length'] = np.zeros((i_e-i_s, 1))
        input['input_length'][:] = img_w // downsample_factor - 2
        #for i in range(input['the_labels'].shape[0]):
        #    print(input['the_labels'][i])
        #print(input['label_length'])
        #print(input['the_input'].shape, input['the_labels'].shape, input['label_length'].shape, input['input_length'].shape)

        outputs = {'ctc': np.zeros([i_e-i_s])}  # dummy data for dummy loss function
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
    test_func = K.function([model.input[0]],
                                  [model.get_layer('softmax').output])
    for n, i in enumerate(batches_list):
        i_s = i * gen.batch_size  # index of the first image in this batch
        i_e = min([(i + 1) * gen.batch_size, gen.val_num])  # index of the last image in this batch
        X, y = gen._data_generation(i_s, i_e, train=False)
        decoded_res = decode_batch(test_func, X['the_input'])
        print(decoded_res)
        preds = model.predict_on_batch(X)
        print(len(preds))
        print(preds[0].shape)

        actual_inds = X['the_labels'].argmax(axis=1)
        print(actual_inds.shape)
        for t_ind in range(X['the_input'].shape[0]):
            box = X['the_input'][t_ind, ...]
            print('predicted', decoded_res[t_ind])
            print('actual', labels_to_text(X['the_labels'][t_ind]))
            cv2.imshow('frame', np.swapaxes(box, 0, 1))
            cv2.waitKey(0)


def convert_output(output):
    intervals = []
    print(output.shape)
    for i in range(output.shape[0]):
        lab = characters[output[i]]
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

def ctc_lambda_func(args):
    ls, y_pred, input_length, label_length = args

    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(ls, y_pred, input_length, label_length)

def labels_to_text(ls):
    ret = []
    for c in ls:
        if c >= len(characters):
            continue
        ret.append(characters[c])
    return ret

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret

class VizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
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
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                ref = [characters[x] for x in word_batch['the_labels'][j] if x < len(characters)]
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
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
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
            pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % ('|'.join([characters[x] for x in word_batch['the_labels'][i] if x < len(characters)]), res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        pylab.close()

if __name__ == '__main__':
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, LSTM, TimeDistributed, Bidirectional, CuDNNGRU, CuDNNLSTM, Conv1D, MaxPooling1D
    from keras.regularizers import l1_l2
    from keras import backend as K
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.layers import Input, Dense, Activation
    from keras.layers import Reshape, Lambda
    from keras.layers.merge import add, concatenate
    from keras.models import Model
    from keras.layers.recurrent import GRU
    from keras.optimizers import SGD
    from keras.utils.data_utils import get_file
    from keras.preprocessing import image
    import keras.callbacks

    run_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    input_shape = (64, 12, 1)
    img_w = 64
    img_h = 12
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
    final_output_weights = os.path.join(working_dir, 'ocr_weights.h5')
    final_output_json = os.path.join(working_dir, 'ocr_model.json')

    output_name = 'output'
    y_train = gen.hdf5_file['train_label_sequence']
    unique, counts = np.unique(y_train, return_counts=True)
    counts = dict(zip(unique, counts))
    label_weights = create_class_weight(counts)
    print(label_weights)

    conv_filters = 32
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    minibatch_size = 32
    downsample_factor = (pool_size ** 2)
    from keras import backend as K

    K.set_learning_phase(1)  # set learning phase
    if not os.path.exists(final_output_json):
        current_model_path = os.path.join(working_dir, 'current_ocr_model.h5')
        if not os.path.exists(current_model_path):
            act = 'relu'
            input_data = Input(name='the_input', shape=input_shape, dtype='float32')
            inner = Conv2D(conv_filters, kernel_size, padding='same',
                           activation=act, #kernel_initializer='he_normal',
                           name='conv1')(input_data)
            inner = Conv2D(conv_filters, kernel_size, padding='same',
                           activation=act, #kernel_initializer='he_normal',
                           name='conv2')(inner)
            inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
            inner = Dropout(0.25)(inner)
            inner = Conv2D(conv_filters*2, kernel_size, padding='same',
                           activation=act, #kernel_initializer='he_normal',
                           name='conv3')(inner)
            inner = Conv2D(conv_filters*2, kernel_size, padding='same',
                           activation=act, #kernel_initializer='he_normal',
                           name='conv4')(inner)
            inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)
            inner = Dropout(0.25)(inner)

            conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters*2)
            inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

            # cuts down input size going into RNN:
            inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

            # Two layers of bidirectional GRUs
            # GRU seems to work as well, if not better than LSTM:
            inner = CuDNNGRU(rnn_size, return_sequences=True)(inner)
            inner = CuDNNGRU(rnn_size, return_sequences=True)(inner)
            #gru_1 = CuDNNGRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
            #gru_1b = CuDNNGRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
            #             name='gru1_b')(
            #    inner)
            #gru1_merged = add([gru_1, gru_1b])
            #gru_2 = CuDNNGRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
            #gru_2b = CuDNNGRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
            #             name='gru2_b')(
            #    gru1_merged)

            # transforms RNN output to character activations:
            inner = Dense(class_count+1, kernel_initializer='he_normal',
                          name='dense2')(inner)
            y_pred = Activation('softmax', name='softmax')(inner)
            Model(inputs=input_data, outputs=y_pred).summary()

            labels_input = Input(name='the_labels', shape=[12], dtype='float32')
            input_length = Input(name='input_length', shape=[1], dtype='uint8')
            label_length = Input(name='label_length', shape=[1], dtype='uint8')
            # Keras doesn't currently support loss funcs with extra parameters
            # so CTC loss is implemented in a lambda layer
            loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
                [labels_input, y_pred, input_length, label_length])

            # clipnorm seems to speeds up convergence
            sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

            model = Model(inputs=[input_data, labels_input, input_length, label_length], outputs=loss_out)

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
        early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0,
                                                      mode='auto')
        tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)

        test_func = K.function([input_data], [y_pred])

        viz_cb = VizCallback(run_name, test_func, validation_generator)
        history = model.fit_generator(generator=training_generator,
                            epochs=num_epochs,
                            steps_per_epoch=gen.data_num // params['batch_size'],
                            # steps_per_epoch=100,
                            validation_data=validation_generator,
                            validation_steps=gen.val_num // params['batch_size'],
                            # validation_steps=100
                            callbacks=[viz_cb, checkpointer, early_stopper, tensorboard],
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