import os
import csv
import keras
from collections import defaultdict
import numpy as np
from keras.preprocessing.text import one_hot
from keras.layers import Embedding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(base_dir, 'raw_data')
data_dir = os.path.join(base_dir, 'data')

results = sorted(['Draw', 'Team1wins', 'Team2wins'])

previous_results = sorted(['Draw', 'Team1wins', 'Team2wins', 'Start'])


def player_name_to_index(name):
    return players.index(name)


def map_name_to_index(name):
    return maps.index(name)


def patch_name_to_index(name):
    return patches.index(name)


def result_to_index(value):
    return results.index(value)


def previous_to_index(value):
    return previous_results.index(value)


def load_collection_file(file_name):
    collection = []
    with open(os.path.join(data_dir, file_name), 'r', encoding='utf8') as f:
        for line in f:
            collection.append(line.strip())
    return collection


players = load_collection_file('players.txt')
maps = load_collection_file('maps.txt')
patches = load_collection_file('patches.txt')
print(patches)


if __name__ == '__main__':
    team_features = ['Team1_player1', 'Team1_player2', 'Team1_player3', 'Team1_player4',
                    'Team1_player5', 'Team1_player6',
                    'Team2_player1', 'Team2_player2', 'Team2_player3', 'Team2_player4',
                    'Team2_player5', 'Team2_player6']
    map_data = []
    patch_data = []
    previous_result_data = []
    batch_size = 32
    num_classes = 3
    epochs = 50
    player_data = defaultdict(list)
    y = []
    with open(os.path.join(data_dir, 'all.txt'), 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            #print(line)
            y.append(result_to_index(line['Result']))
            for tf in team_features:
                player_data[tf].append(player_name_to_index(line[tf]))
            map_data.append(map_name_to_index(line['Map']))
            patch_data.append(patch_name_to_index(line['Patch']))
            previous_result_data.append(previous_to_index(line['PreviousResult']))
    #test_size = int(len(y) * 0.2)
    #test_indices = np.random.choice(len(y), test_size)
    #train_indices = [x for x in range(len(y)) if x not in test_indices
    #team1_player1s_train = team1_player1s[train_indices]
    #team1_player1s_test = team1_player1s[test_indices]

    #team1_player2s_train = team1_player2s[train_indices]
    #team1_player2s_test = team1_player2s[test_indices]]
    for k, v in player_data.items():
        player_data[k] = keras.utils.to_categorical(np.array(v), len(players))

    map_data = keras.utils.to_categorical(np.array(map_data), len(maps))
    patch_data = keras.utils.to_categorical(np.array(patch_data), len(patches))
    previous_result_data = keras.utils.to_categorical(np.array(previous_result_data), len(previous_results))

    y = keras.utils.to_categorical(np.array(y), len(results))
    #y_train = y[train_indices]
    #y_test = y[test_indices]

    #print(team1_player1s_train.shape, y_train.shape)
    #print(team1_player1s_test.shape, y_test.shape)
    player_shape = len(players)

    team_inputs = {}
    for k, v in player_data.items():
        name = k+'_input'
        team_inputs[name]= Input(shape=(player_shape,), name=name)

    map_input = Input(shape=(len(maps),), name = 'map_input')
    patch_input = Input(shape=(len(patches),), name = 'patch_input')
    print(patch_data.shape)
    shared_player = Dense(32, activation='tanh')
    shared_map = Dense(8, activation='tanh')
    shared_patch = Dense(8, activation='tanh')
    team_encodeds = {}
    for k, v in team_inputs.items():
        team_encodeds[k] = shared_player(v)
        #team_encodeds[k] = Dropout(0.2)(team_encodeds[k])
        #team_encodeds[k] = Dense(512, activation='tanh')(team_encodeds[k])
        #team_encodeds[k] = Dropout(0.2)(team_encodeds[k])
        #team_encodeds[k] = Dense(512, activation='tanh')(team_encodeds[k])

    encoded_map = shared_map(map_input)
    encoded_patch = shared_patch(patch_input)
    previous_input = Input(shape=(len(previous_results),), name='previous_input')
    x = keras.layers.concatenate(list(team_encodeds.values()) + [encoded_map, encoded_patch], axis=-1)

    x = Dense(512, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='tanh')(x)
    x = Dropout(0.)(x)

    output = Dense(num_classes, activation='softmax', name='main_output')(x)

    model = Model(inputs=list(team_inputs.values()) + [map_input, patch_input, previous_input], outputs=output)

    model.summary()

    model.compile(loss={'main_output':'categorical_crossentropy'},
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    fit_data = {k: player_data[k.replace('_input', '')] for k in team_inputs.keys()}
    fit_data['map_input'] = map_data
    fit_data['patch_input'] = patch_data
    fit_data['previous_input'] = previous_result_data
    history = model.fit(fit_data,
              {'main_output': y},
              epochs=epochs, batch_size=batch_size, validation_split=0.1)
    #score = model.evaluate([team1_player1s_test, team1_player2s_test], y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    from keras.utils import plot_model

    plot_model(model, to_file=os.path.join(data_dir, 'model.png'))
