import os
import csv
import cv2
import math
import random
import numpy as np
import h5py

working_dir = r'E:\Data\Overwatch\models\kf_lstm'
os.makedirs(working_dir, exist_ok=True)

train_dir = r'E:\Data\Overwatch\training_data\kf_lstm'
hdf5_path = os.path.join(train_dir, 'dataset.hdf5')
slot_based = False
set_files = {}
sets = {}
set_files['first_hero'] = os.path.join(train_dir, 'hero_set.txt')
set_files['first_color'] = os.path.join(train_dir, 'color_set.txt')
set_files['headshot'] = os.path.join(train_dir, 'headshot_set.txt')
set_files['ability'] = os.path.join(train_dir, 'ability_set.txt')
set_files['second_hero'] = os.path.join(train_dir, 'hero_set.txt')
set_files['second_color'] = os.path.join(train_dir, 'color_set.txt')


def load_set(path):
    ts = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            ts.append(line.strip())
    return ts


for k, v in set_files.items():
    sets[k] = load_set(v)

class_counts = {}
for k, v in sets.items():
    class_counts[k] = len(v)


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
        if slot_based:
            self.data_num = self.hdf5_file["train_img"].shape[0]
            self.val_num = self.hdf5_file["val_img"].shape[0]
            print(self.hdf5_file["train_img"].shape)
        else:
            self.data_num = self.hdf5_file["train_slot_0_img"].shape[0]
            self.val_num = self.hdf5_file["val_slot_0_img"].shape[0]
        self.subtract_mean = subtract_mean
        if self.subtract_mean:
            self.mm = self.hdf5_file["train_mean"][0, ...]
            self.mm = self.mm[np.newaxis, ...].astype(np.uint8)

    def __get_exploration_order(self, train=True):
        'Generates order of exploration'
        # Find exploration order
        if train:
            batches_list = list(range(int(math.ceil(float(self.data_num) / self.batch_size))))
        else:
            batches_list = list(range(int(math.ceil(float(self.val_num) / self.batch_size))))
        if self.shuffle:
            random.shuffle(batches_list)
        return batches_list

    def __data_generation(self, i_s, i_e, train=True):
        'Generates data of batch_size samples'  # X : (n_samples, v_size, v_size, v_size, n_channels)
        if train:
            pre = 'train'
        else:
            pre = 'val'

        if slot_based:
            outputs = []
            inputs = []
            for slot in range(6):
                inputs.append(self.hdf5_file["{}_slot_{}_img".format(pre, slot)][i_s:i_e, ...])
                # print(hero_set[self.hdf5_file["{}_hero_label".format(pre)][i_s]])
                # cv2.imshow('frame', images[0, :])
                # cv2.waitKey(0)
                output = {}
                for k, count in class_counts.items():
                    output['{}_output'.format(k)] = sparsify(self.hdf5_file["{}_{}_label".format(pre, k)][i_s:i_e, ...], count)
                outputs.append(output)
            return inputs, outputs
        else:
            output = {}
            input = {}
            input['dummy_input'] = np.zeros((i_e - i_s, 100, 50))
            for slot in range(6):
                input['slot_{}_input'.format(slot)] =  self.hdf5_file["{}_slot_{}_img".format(pre, slot)][i_s:i_e, ...]
                for k, count in class_counts.items():
                    output['slot_{}_{}_output'.format(slot, k)] = sparsify(self.hdf5_file["{}_slot_{}_{}_label".format(pre, slot, k)][i_s:i_e, ...], count)
        return input, output

    def generate_train(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            batches_list = self.__get_exploration_order()

            for n, i in enumerate(batches_list):
                i_s = i * self.batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * self.batch_size, self.data_num])  # index of the last image in this batch
                X, y = self.__data_generation(i_s, i_e)
                if slot_based:
                    for slot in range(6):
                        yield X[slot], y[slot]
                else:
                    yield X, y

    def generate_val(self):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            batches_list = self.__get_exploration_order(train=False)

            for n, i in enumerate(batches_list):
                i_s = i * self.batch_size  # index of the first image in this batch
                i_e = min([(i + 1) * self.batch_size, self.val_num])  # index of the last image in this batch
                X, y = self.__data_generation(i_s, i_e, train=False)
                if slot_based:
                    for slot in range(6):
                        yield X[slot], y[slot]
                else:
                    yield X, y

def create_class_weight(labels_dict,mu=1):
    total = np.sum(np.array(list(labels_dict.values())))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight

if __name__ == '__main__':

    input_shape = (100, 50)
    params = {'dim_x': 67,
              'dim_y': 67,
              'dim_z': 3,
              'batch_size': 10,
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
    current_model_path = os.path.join(working_dir, 'current_kf_model.hdf5')
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input, LSTM, TimeDistributed, Bidirectional, Conv1D, MaxPooling1D, GRU, CuDNNGRU, CuDNNLSTM
    if not os.path.exists(current_model_path):
        if slot_based:
            input = Input(shape=(100, 50))
            #shared_lstm = CuDNNGRU(64, return_sequences=True)(input)
            #shared_lstm = CuDNNGRU(64, return_sequences=True)(shared_lstm)
            #out = CuDNNGRU(64, return_sequences=True)(shared_lstm)
            x = Bidirectional(GRU(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.2))(input)
            x = Bidirectional(GRU(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.2))(x)
            x = Bidirectional(GRU(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.2))(x)
            outputs = []

            for k, count in class_counts.items():
                outputs.append(TimeDistributed(Dense(count, activation='softmax'), name='{}_output'.format(k))(x))
            model = Model(input=input, outputs=outputs)
            model.summary()
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
        else:
            input = Input(shape=(100, 100))
            #shared_lstm = CuDNNGRU(64, return_sequences=True)(input)
            #shared_lstm = CuDNNGRU(64, return_sequences=True)(shared_lstm)
            #out = CuDNNGRU(64, return_sequences=True)(shared_lstm)
            shared_lstm = Bidirectional(GRU(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.2))(input)
            shared_lstm = Bidirectional(GRU(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.2))(shared_lstm)
            out = Bidirectional(GRU(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.2))(shared_lstm)
            time_model = Model(inputs=[input], outputs=[out])
            inputs = []
            outputs = []
            inputs.append(Input(shape=input_shape, name='dummy_input'))

            for slot in range(6):
                inputs.append(Input(shape=input_shape, name='slot_{}_input'.format(slot)))
                if slot == 0:
                    merged_vector = keras.layers.concatenate([inputs[0], inputs[-1]], axis=-1)
                else:
                    merged_vector = keras.layers.concatenate([inputs[-2], inputs[-1]], axis=-1)
                x = time_model(merged_vector)
                for k, count in class_counts.items():
                    outputs.append(TimeDistributed(Dense(count, activation='softmax'), name='slot_{}_{}_output'.format(slot, k))(x))

            model = Model(inputs=inputs, outputs=outputs)
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
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=0,
                                                  mode='auto')
    print(gen.val_num // params['batch_size'])
    history = model.fit_generator(generator=training_generator,
                                  epochs=num_epochs,
                                  steps_per_epoch=gen.data_num // params['batch_size'],
                                  # steps_per_epoch=100,
                                  validation_data=validation_generator,
                                  validation_steps=gen.val_num // params['batch_size'],
                                  # validation_steps=100
                                  callbacks=[checkpointer, early_stopper]
                                  )
    final_output_weights = os.path.join(working_dir, 'kf_weights.h5')
    final_output_json = os.path.join(working_dir, 'kf_model.json')
    model.save_weights(final_output_weights)
    model_json = model.to_json()
    with open(final_output_json, "w") as json_file:
        json_file.write(model_json)
    # list all data in history
    print(history.history.keys())
    import matplotlib.pyplot as plt

    # summarize history for accuracy
    plt.plot(history.history['slot_0_second_hero_output_acc'])
    plt.plot(history.history['val_slot_0_second_hero_output_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
