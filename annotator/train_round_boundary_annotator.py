import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
import numpy as np

working_dir = r'E:\Data\Overwatch\models'
os.makedirs(working_dir, exist_ok=True)

data_dir = r'E:\Data\Overwatch\raw_data'
train_dir = r'C:\Users\micha\Documents\Data\cnn_train'
annotations_dir = os.path.join(data_dir, 'annotations')
test_files = ['513', '1633', '1881', '1859', '1852', '1853', '1858']

actual_starts = {
    '1852': 880,
    '1853': 628,
}

have_correct_data = ['513', '1881', '1859', '1858', '1853', '1852']

output_categories = ['MATCH_NOT_STARTED', 'MATCH_FINISHED', 'IN_GAME', 'PAUSED', 'NOT_IN_GAME']
cnn_output_categories = ['IN_GAME', 'NOT_IN_GAME']

label_file = os.path.join(train_dir, 'labels.npy')


def load_events(match_dir, fps, actual_start=None):
    events = []
    with open(os.path.join(match_dir, 'events.txt'), 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            time, event = line.split('\t')
            time = int(time)
            frame = int(time * fps)
            if event not in ['MATCH', 'END', 'PAUSE', 'UNPAUSE']:
                continue
            events.append([frame, event])
    diff = events[0][0] - actual_start
    for i in range(len(events)):
        events[i][0] -= diff
    return events


def get_label(frame, events):
    if frame < events[0][0]:
        return 'NOT_IN_GAME'
    if frame >= events[-1][0]:
        return 'NOT_IN_GAME'
    for i, e in enumerate(events):
        if frame >= e[0] and frame < events[i + 1][0]:
            if e[1] in ['MATCH', 'UNPAUSE']:
                return 'IN_GAME'
            elif e[1] == 'PAUSE':
                return 'IN_GAME'
            else:
                return 'NOT_IN_GAME'
    print(frame)
    print(events)
    error


def generate_train_data():
    os.makedirs(train_dir, exist_ok=True)
    labels = []
    frame_id = 0
    for m in have_correct_data:
        print(m)
        match_dir = os.path.join(annotations_dir, m)
        mid_file = os.path.join(match_dir, 'mid.avi')
        print(mid_file)
        cap = cv2.VideoCapture(mid_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        actual_start = None
        if m in actual_starts:
            actual_start = actual_starts[m]
        events = load_events(match_dir, fps, actual_start)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(num_frames)
        frame_count = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            label = get_label(frame_count, events)
            np.save(os.path.join(train_dir, '{}.npy'.format(frame_id)), frame)
            frame_count += 1
            frame_id += 1
            labels.append(cnn_output_categories.index(label))
        cap.release()
    np.save(label_file, np.array(labels))


def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = 2  # Enter number of classes
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])


class DataGenerator(object):
    def __init__(self, dim_x=140, dim_y=300, dim_z=3, batch_size=32, shuffle=True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, labels, list_IDs_temp):
        'Generates data of batch_size samples'  # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            X[i, :, :, :] = np.load(os.path.join(train_dir, '{}.npy'.format(ID)))
            # Store class
            y[i] = labels[ID]

        return X, sparsify(y)

    def generate(self, labels, list_IDs):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                # Generate data
                X, y = self.__data_generation(labels, list_IDs_temp)

                yield X, y


if __name__ == '__main__':
    if True or not os.path.exists(train_dir):
        generate_train_data()
    # Parameters
    num_classes = len(cnn_output_categories)
    params = {'dim_x': 140,
              'dim_y': 300,
              'dim_z': 3,
              'batch_size': 32,
              'shuffle': True}

    # Datasets
    labels = np.load(label_file)  # Labels
    num_files = labels.shape[0]
    train_size = int(num_files * 0.95)
    import random

    partition = {'train': range(train_size)}  # IDs
    partition['validation'] = range(train_size, num_files)

    # Generators
    training_generator = DataGenerator(**params).generate(labels, partition['train'])
    validation_generator = DataGenerator(**params).generate(labels, partition['validation'])
    print('set up complete')
    # Design model
    input_shape = (140, 300, 3)
    model = Sequential()
    # model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(140, 300, 3)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print('model compiled')
    # Train model on dataset
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(working_dir, 'model.{epoch:02d}-{val_loss:.2f}.hdf5'), verbose=1, save_best_only=True)
    model.fit_generator(generator=training_generator,
                        epochs=20,
                        steps_per_epoch=len(partition['train']) // params['batch_size'],
                        # steps_per_epoch=100,
                        validation_data=validation_generator,
                        validation_steps=len(partition['validation']) // params['batch_size'],
                        # validation_steps=100
                        callbacks=[checkpointer]
                        )
    model.save(os.path.join(working_dir, 'model.h5'))
