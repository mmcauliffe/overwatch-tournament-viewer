import annotator.config as config
import requests
import numpy as np
import os


def get_hero_list():
    url = config.api_url + 'heroes/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_player_list():
    url = config.api_url + 'train_players/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_color_list():
    url = config.api_url + 'team_colors/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_spectator_modes():
    url = config.api_url + 'spectator_mode_choices/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_status_types():
    url = config.api_url + 'status_effect_choices/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_maps():
    url = config.api_url + 'maps/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_npc_list():
    url = config.api_url + 'npcs/'
    r = requests.get(url)
    return sorted(set(x['name'].lower() for x in r.json()))


def get_map_modes():
    url = config.api_url + 'map_modes/'
    r = requests.get(url)
    return sorted(set(x.lower() for x in r.json()))


def get_ability_list():
    ability_set = set()
    url = config.api_url + 'abilities/damaging_abilities/'
    r = requests.get(url)
    resp = r.json()
    for a in resp:
        ability_set.add(a['name'].lower())
        if a['headshot_capable']:
            ability_set.add(a['name'].lower() + ' headshot')
    url = config.api_url + 'abilities/reviving_abilities/'
    r = requests.get(url)
    resp = r.json()
    for a in resp:
        ability_set.add(a['name'].lower())
    ability_set.add('defense matrix')
    return ability_set


def get_train_info():
    url = config.api_url + 'train_info/'
    r = requests.get(url)
    data = r.json()
    for k, v in data.items():
        print(k, v)
    return data


def get_train_rounds():
    url = config.api_url + 'train_rounds/'
    r = requests.get(url)
    rounds = r.json()
    print('Total number of rounds:', len(rounds))
    return rounds


def get_example_rounds():
    url = config.api_url + 'example_rounds/'
    r = requests.get(url)
    rounds = r.json()
    print('Total number of rounds:', len(rounds))
    return rounds


def get_train_rounds_plus():
    url = config.api_url + 'train_rounds_plus/'
    r = requests.get(url)
    return r.json()


def get_train_vods():
    url = config.api_url + 'train_vods/'
    r = requests.get(url)
    return r.json()


def get_annotate_rounds():
    url = config.api_url + 'annotate_rounds/'
    r = requests.get(url)
    return r.json()


def get_annotate_vods_in_out_game():
    url = config.api_url + 'annotate_vods/in_out_game/'
    r = requests.get(url)
    return r.json()


def get_annotate_vods_round_events():
    url = config.api_url + 'annotate_vods/round_events/'
    r = requests.get(url)
    return r.json()


def load_token():
    directory = os.path.dirname(os.path.abspath(__file__))
    if config.DEV:
        path = os.path.join(directory,'auth_token_dev')
    else:
        path = os.path.join(directory,'auth_token')
    with open(path, 'r') as f:
        token = f.read().strip()
    return token


def upload_annotated_in_out_game(data):
    url = config.api_url + 'annotate_vods/upload_in_out_game/'
    r = requests.post(url, json=data, headers={'Authorization': 'Token {}'.format(load_token())})
    return r.json()


def upload_annotated_round_events(data):
    url = config.api_url + 'annotate_rounds/{}/'.format(data['round'])
    r = requests.put(url, json=data, headers={'Authorization': 'Token {}'.format(load_token())})
    return r.json()


def get_player_states(round_id):
    from annotator.game_values import STATUS_SET
    url = config.api_url + 'rounds/{}/player_states/'.format(round_id)
    r = requests.get(url)
    data = r.json()
    for side, d in data.items():
        for ind, v in d.items():
            for k in ['ult', 'alive', 'hero'] + STATUS_SET:
                if k == '':
                    continue
                data[side][ind]['{}_array'.format(k)] = np.array([x['end'] for x in v[k]])
    return data


def get_round_states(round_id):
    url = config.api_url + 'rounds/{}/round_states/'.format(round_id)
    r = requests.get(url)
    return r.json()


def get_kf_events(round_id):
    url = config.api_url + 'rounds/{}/kill_feed_events/'.format(round_id)
    r = requests.get(url)
    events = r.json()
    for e in events:
        for k, v in e.items():
            if isinstance(v, str):
                e[k] = v.lower()
            elif isinstance(v, list):
                e[k] = [x.lower() for x in v]
    return events


def upload_game(data):
    url = config.api_url + 'games/'
    print(data)
    resp = requests.post(url, json=data)
    if resp.status_code != 200:
        raise Exception
    print(resp)


def update_annotations(data, round_id):
    to_send = {'player_states': {},
               'kill_feed': data['kill_feed'],
               'replays': data['replays'],
               'pauses': data['pauses'],
               'ignore_switches': data['ignore_switches']
               }
    for k, v in data['player'].items():
        k = '{}_{}'.format(*k)
        to_send['player_states'][k] = {}
        switches = v.generate_switches()
        ug, uu = v.generate_ults()

        to_send['player_states'][k]['switches'] = switches
        to_send['player_states'][k]['ult_gains'] = ug
        to_send['player_states'][k]['ult_uses'] = uu
    url = config.api_url + 'annotate_rounds/{}/'.format(round_id)
    print(to_send)
    resp = requests.put(url, json=to_send)
    if resp.status_code != 200:
        raise Exception
    print(resp)

