import os
import csv
import datetime
from collections import Counter

base_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(base_dir, 'raw_data')
data_dir = os.path.join(base_dir, 'data')


min_matches = 0

def load_collection_file(file_name):
    collection = []
    with open(os.path.join(raw_data_dir, file_name), 'r', encoding='utf8') as f:
        for line in f:
            collection.append(line.strip())
    return collection


def load_csv_file(file_name):
    collection = []
    with open(os.path.join(raw_data_dir, file_name), 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for line in reader:
            collection.append(line)
    return collection


def get_players():
    return load_collection_file('players.txt')


def get_patches():
    patches = {}
    for p in load_csv_file('patches.txt'):
        patches[p['PatchName']] = {'Begin': datetime.datetime.strptime(p['Begin'], '%Y-%m-%d'),
                                   'End': datetime.datetime.strptime(p['End'], '%Y-%m-%d'),
                                   'Length': int(p['Length'])}
    return patches


def get_matches():
    collection = load_csv_file('matches.txt')
    for m in collection:
        m['Date'] = datetime.datetime.strptime(m['Date'], '%Y-%m-%d')
    return collection


def get_picks():
    collection = load_csv_file('hero_pick_rate.txt')
    for m in collection:
        m['Date'] = datetime.datetime.strptime(m['Date'], '%Y-%m-%d')
    return collection


def get_maps():
    return load_collection_file('maps.txt')


def get_teams():
    return load_collection_file('teams.txt')


def get_heroes():
    return load_collection_file('heroes.txt')


def figure_patch(date):
    for patch_name, data in patches.items():
        if data['Begin'] <= date < data['End']:
            return patch_name


def preprocess_data(full_data_only=True):
    patch_counts = Counter()
    player_counts = Counter()
    map_counts = Counter()
    match_data_set = []
    index = {}
    for m in matches:
        players = [v for k, v in m.items() if 'player' in k]
        patch = figure_patch(m['Date'])
        patch_counts.update([patch])
        player_counts.update(players)
        map_counts.update([m['Map']])

        patch_progress = (m['Date'] - patches[patch]['Begin']) / (patches[patch]['End'] - patches[patch]['Begin'])
        m['PatchProgress'] = patch_progress
        m['Patch'] = patch
        index[m['WinstonLabID']] = m
    print(patch_counts)
    print(player_counts)
    print(map_counts)
    unknown_players = [k for k, v in player_counts.items() if v < min_matches]
    unknown_counts = sum([v for k, v in player_counts.items() if v < min_matches])
    print(len(unknown_players), unknown_counts)
    full_data = []
    all_data = []
    pick_data = []
    players = set()
    for m in matches:
        row = {k: v for k, v in m.items() if k in match_header}
        for k, v in row.items():
            if v in unknown_players:
                row[k] = 'unknown_player'
        players.update(v for k, v in row.items() if 'player' in k)
        if m['FullData'] == 'True':
            full_data.append(row)
        all_data.append(row)

    hero_picks = Counter()
    for p in picks:
        # match = index[p['WinstonLabID']]
        patch = figure_patch(p['Date'])
        patch_progress = (p['Date'] - patches[patch]['Begin']) / (patches[patch]['End'] - patches[patch]['Begin'])
        row = {k: v for k, v in p.items() if k in pick_header}
        row['Patch'] = patch
        row['PatchProgress'] = patch_progress
        pick_data.append(row)
        for h in heroes:
            hero_picks[h] += float(p[h])

    for k, v in hero_picks.items():
        hero_picks[k] = v / len(picks)
    print(hero_picks)

    output_clean(full_data, all_data, pick_data)
    with open(os.path.join(data_dir, 'players.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted(players)))
    with open(os.path.join(data_dir, 'patches.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(sorted(patches.keys())))



def output_clean(full_data, all_data, pick_data):
    with open(os.path.join(data_dir, 'full.txt'), 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, match_header)
        writer.writeheader()
        for r in full_data:
            writer.writerow(r)

    with open(os.path.join(data_dir, 'all.txt'), 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, match_header)
        writer.writeheader()
        for r in all_data:
            writer.writerow(r)

    with open(os.path.join(data_dir, 'picks.txt'), 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, pick_header)
        writer.writeheader()
        for r in pick_data:
            writer.writerow(r)


def basic_analysis():
    print(matches[:10])
    print(patches)
    preprocess_data()


if __name__ == '__main__':
    players = get_players()
    teams = get_teams()
    maps = get_maps()
    heroes = get_heroes()
    patches = get_patches()
    matches = get_matches()
    picks = get_picks()
    match_header = ['Result', 'Map', 'PreviousResult', 'Patch', 'PatchProgress',
                    'Team1_player1', 'Team1_player2', 'Team1_player3', 'Team1_player4',
                    'Team1_player5', 'Team1_player6',
                    'Team2_player1', 'Team2_player2', 'Team2_player3', 'Team2_player4',
                    'Team2_player5', 'Team2_player6']
    pick_header = ['Player', 'Map', 'Patch', 'PatchProgress'] + heroes
    basic_analysis()
