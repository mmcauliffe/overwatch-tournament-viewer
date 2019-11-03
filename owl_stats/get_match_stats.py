import requests
import os
import json
from pprint import PrettyPrinter
from datetime import datetime, date

printer = PrettyPrinter(indent=4)

data_directory = r'E:\Data\Overwatch\owl_stats\owl_api'

os.makedirs(data_directory, exist_ok=True)

map_mapping = {}

def get_maps():
    resp = requests.get('https://api.overwatchleague.com/maps')
    data = resp.json()
    print(data)
    for m in data:
        map_mapping[m['guid']] = m['name']['en_US']


def get_matches():
    resp = requests.get('https://api.overwatchleague.com/matches')
    data = resp.json()
    return data['content']

def get_previously_done_matches():
    prev = []
    for file_name in os.listdir(data_directory):
        with open(os.path.join(data_directory, file_name)) as f:
            d = json.load(f)
            prev.append(d['id'])
    return prev

def get_stats(match):
    team_mapping = {}
    for t in match['competitors']:
        team_mapping[t['id']] = {'name': t['name'], 'players': {}}
        for player in t['players']:
            print(player)
            team_mapping[player['team']['id']]['players'][player['player']['id']] = player['player']['name'].lower()
            print(team_mapping[player['team']['id']])
        #printer.pprint(data)
    #printer.pprint(match)
    team_one, team_two = list(team_mapping.values())
    match_data = {'id': match['id'], 'date': str(datetime.fromtimestamp(int(match['startDate']/1000)).date()),
                  'games': [], 'team_one': team_one['name'], 'team_two':team_two['name']}
    for g in match['games']:
        print(match['id'], g['number'])
        print('GETTING PLAYERS')
        for player in g['players']:
            print(player)
            team_mapping[player['team']['id']]['players'][player['player']['id']] = player['player']['name'].lower()
            print(team_mapping[player['team']['id']])
        resp = requests.get('https://api.overwatchleague.com/stats/matches/{}/maps/{}'.format(match['id'], g['number']))
        if resp.status_code == 404:
            continue
        data = resp.json()
        #printer.pprint(data)
        game_data = {'map': map_mapping[data['map_id']],
                     'match_id': data['esports_match_id'],
                     'game_number': data['game_number']}

        for i, t in enumerate(data['teams']):
            team = team_mapping[t['esports_team_id']]
            print(team)
            team_data = {'name': team['name'], 'stats': []}
            for x in t['players']:
                print(x)
                d = {'name': team['players'][x['esports_player_id']], 'hero_stats':{}}
                for s in x['stats']:
                    d[s['name']] = s['value']
                for h in x['heroes']:
                    d['hero_stats'][h['name']] = {}
                    for s in h['stats']:
                        d['hero_stats'][h['name']][s['name']] = s['value']
                team_data['stats'].append(d)
            game_data['team_{}'.format(i)] = team_data
        match_data['games'].append(game_data)
    filename = '{} - {} vs {}.json'.format(match_data['date'], match_data['team_one'], match_data['team_two'])
    with open(os.path.join(data_directory, filename), 'w') as f:
        json.dump(match_data, f, indent=4)
    #error

if __name__ == '__main__':
    maps = get_maps()
    prev = get_previously_done_matches()
    matches = get_matches()
    for m in matches:
        if m['id'] in prev:
            continue
        print(m['id'])
        get_stats(m)