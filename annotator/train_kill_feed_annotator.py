import os
import csv
import cv2
import numpy as np
import h5py
import math
import random

working_dir = r'E:\Data\Overwatch\models\kf_cnn'
os.makedirs(working_dir, exist_ok=True)

train_dir = r'C:\Users\micha\Documents\Data\kf_cnn'
train_dir = r'E:\Data\Overwatch\training_data\kf_cnn'
hdf5_path = os.path.join(train_dir, 'dataset.hdf5')

hero_set_file = os.path.join(train_dir, 'hero_set.txt')
color_set_file = os.path.join(train_dir, 'color_set.txt')
ability_set_file = os.path.join(train_dir, 'ability_set.txt')
headshot_set_file = os.path.join(train_dir, 'headshot_set.txt')


set_files = {
    'first_hero': os.path.join(train_dir, 'hero_set.txt'),
    'first_color': os.path.join(train_dir, 'color_set.txt'),
    'headshot': os.path.join(train_dir, 'headshot_set.txt'),
    'ability': os.path.join(train_dir, 'ability_set.txt'),
    'second_hero': os.path.join(train_dir, 'hero_set.txt'),
    'second_color': os.path.join(train_dir, 'color_set.txt'),

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

def sparsify(y, n_classes):
    'Returns labels in binary NumPy array'
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])


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
        images = self.hdf5_file["{}_img".format(pre)][i_s:i_e, ...]
        if self.subtract_mean:
            images -= self.mm
        output = {}
        for k, count in class_counts.items():
            output['{}_output'.format(k)] = sparsify(self.hdf5_file["{}_{}_label".format(pre, k)][i_s:i_e], count)
        return images, output

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
                yield X, y


if __name__ == '__main__':
    import keras
    from keras.models import Sequential, Model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Input

    input_shape = (32, 210, 3)
    params = {'dim_x': 32,
              'dim_y': 210,
              'dim_z': 3,
              'batch_size': 32,
              'shuffle': True,
              'subtract_mean': False}
    num_epochs = 20
    # Datasets

    # Generators
    gen = DataGenerator(hdf5_path, **params)
    training_generator = gen.generate_train()
    validation_generator = gen.generate_val()
    print('set up complete')
    # Design model
    current_model_path = os.path.join(working_dir, 'current_kf_model.h5')
    if not os.path.exists(current_model_path):
        main_input = Input(shape=input_shape, name='main_input')
        x = Conv2D(64, kernel_size=(3, 3),
                   activation='relu')(main_input)
        x = Conv2D(64, kernel_size=(3, 3),
                   activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(50, activation='relu', name='representation')(x)
        x = Dropout(0.5)(x)

        outputs = []
        for k, count in class_counts.items():
            outputs.append(Dense(count, activation='softmax', name=k+'_output')(x))

        model = Model(inputs=[main_input], outputs=outputs)
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
    history = model.fit_generator(generator=training_generator,
                        epochs=num_epochs,
                        steps_per_epoch=gen.data_num // params['batch_size'],
                        # steps_per_epoch=100,
                        validation_data=validation_generator,
                        validation_steps=gen.val_num // params['batch_size'],
                        # validation_steps=100
                        callbacks=[checkpointer, early_stopper]
                        )
    final_output_weights = os.path.join(working_dir, 'player_weights.h5')
    final_output_json = os.path.join(working_dir, 'player_model.json')
    model.save_weights(final_output_weights)
    model_json = model.to_json()
    with open(final_output_json, "w") as json_file:
        json_file.write(model_json)
    # list all data in history
    print(history.history.keys())
    import matplotlib.pyplot as plt
    # summarize history for accuracy
    plt.plot(history.history['first_hero_output_acc'])
    plt.plot(history.history['val_first_hero_output_acc'])
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
