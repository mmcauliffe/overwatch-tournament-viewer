import requests
from bs4 import BeautifulSoup
import os
import sys
import re
import csv
import json
import datetime
import calendar
import time

# base_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = r'E:\Data\Overwatch'
raw_data_dir = os.path.join(base_dir, 'raw_data')
data_dir = os.path.join(base_dir, 'data')
annotations_dir = os.path.join(raw_data_dir, 'annotations')

os.makedirs(annotations_dir, exist_ok=True)

heroes = set()
teams = set()

maps = set()

patches = set()

players = set()

id = 513

match_template = 'https://www.winstonslab.com/matches/match.php?id={}'
event_template = 'https://www.winstonslab.com/events/event.php?id={}'

test_files = [513, 1633, 1881, 1859, 1852, 1853, 1858]

class NotFoundID(Exception):
    pass


def get_team_name(soup, team_id):
    team = soup.find_all('div', class_='team{}-name'.format(team_id))
    team_name = team[0].find_all('a')[0].get_text().lower()
    team_id = re.search(r'id=(\d+)', team[0].find_all('a')[0]['href']).groups()[0]
    return team_name, team_id


def get_players(soup, class_name, team_name):
    team_table = soup.find_all('table', class_=class_name)
    if team_table:
        cells = team_table[0].find_all('td')
        team_players = []
        for c in cells:
            link = c.find_all('a')
            if not link:
                continue
            team_players.append(link[0].get_text().lower())
    else:
        return []
    return sorted(team_players)


def get_event(soup):
    link = soup.find_all('a', class_='match-event')[0]
    m = re.search(r'id=(\d+)', link['href'])
    id = m.groups()[0]
    name = link.get_text().lower()
    return name, id


def get_players_from_event_page(soup, team_name):
    link = soup.find_all('a', class_='match-event')[0]
    m = re.search(r'id=(\d+)', link['href'])
    id = m.groups()[0]
    page = requests.get(event_template.format(id))
    s = BeautifulSoup(page.content, 'html.parser')
    teams = s.find_all('div', class_='participant')
    for t in teams:
        team = t.find('center').find('b').find('a').get_text().lower()
        if team != team_name:
            continue
        players = [x['title'].lower() for x in t.find_all('div', class_='participant-player')]
        return players


def get_maps_and_scores(soup, flip):
    global maps

    try:
        map_scores = soup.find('div', class_='mini-map-scores').find_all('div', class_='map-score')
    except AttributeError:
        return None
    map_wins = []
    for ms in map_scores:
        map_name = ms.find('div', class_='mapname')['title'].lower()
        maps.add(map_name)
        t = ms.find('div', class_='score1')
        t2 = ms.find('div', class_='score2')
        if flip:
            if 'winner' in t['class']:
                result = 'Team2wins'
            elif 'winner' in t2['class']:
                result = 'Team1wins'
            else:
                result = 'Draw'
        else:
            if 'winner' in t['class']:
                result = 'Team1wins'
            elif 'winner' in t2['class']:
                result = 'Team2wins'
            else:
                result = 'Draw'
        map_wins.append((map_name, result))
                # score1 = [x.get_text() for x in ms.find('div', class_='score1').find_all('span')]
                # score2 = [x.get_text() for x in ms.find('div', class_='score2').find_all('span')]
                # scores.append((score1, score2))
    return map_wins


def get_date(soup):
    date_span = soup.find('span', id="tzDate_1")
    date_text = date_span.get_text()
    m = re.match('(\d+)\w+\sof\s(\w+)\s(\d+)', date_text)
    day, month_name, year = m.groups()
    date = datetime.date(year=int(year), day=int(day), month=list(calendar.month_abbr).index(month_name[:3]))
    return date


def check_exists(soup):
    not_found = soup.find_all('span', class_='label-danger')
    if not_found:
        raise (NotFoundID())


def get_vod_link(soup):
    twitch_div = soup.find('a', class_='stream')
    if twitch_div is None:
        return None
    return twitch_div['href']


def parse_page(id):
    match_dir = os.path.join(annotations_dir, str(id))
    if os.path.exists(match_dir):
        return None, None
    global teams
    global players
    data = []
    hero_pick_data = []
    page = requests.get(match_template.format(id))

    soup = BeautifulSoup(page.content, 'html.parser')
    check_exists(soup)
    date = get_date(soup)
    today = date.today()
    if date >= today:
        print("Match hasn't happened yet")
        return None, None
    os.makedirs(match_dir, exist_ok=True)
    team1_name, team1_id = get_team_name(soup, '1')
    team2_name, team2_id = get_team_name(soup, '2')
    team1_players = get_players(soup, 'left-side', team1_name)
    team2_players = get_players(soup, 'right-side', team2_name)
    full_data = True
    if not team1_players:
        team1_players = get_players_from_event_page(soup, team1_name)
        team2_players = get_players_from_event_page(soup, team2_name)
        full_data = False
    teams.add(team1_name)
    teams.add(team2_name)
    flip = team2_name < team1_name
    if flip:
        team1_name, team2_name = team2_name, team1_name
        team1_players, team2_players = team2_players, team1_players
        team1_id, team2_id = team2_id, team1_id
    map_wins = get_maps_and_scores(soup, flip)
    if map_wins is None:
        return None, None
    hero_picks = get_hero_pick_rates(soup)
    event, event_id = get_event(soup)
    base_row = {'Date': str(date), 'Team1': team1_name, 'Team2': team2_name,
                'WinstonLabID': id, 'FullData': full_data, 'EventName': event, 'EventID': event_id,
                'Team1_player1': team1_players[0], 'Team1_player2': team1_players[1], 'Team1_player3': team1_players[2],
                'Team1_player4': team1_players[3], 'Team1_player5': team1_players[4], 'Team1_player6': team1_players[5],
                'Team2_player1': team2_players[0], 'Team2_player2': team2_players[1], 'Team2_player3': team2_players[2],
                'Team2_player4': team2_players[3], 'Team2_player5': team2_players[4], 'Team2_player6': team2_players[5]
                }
    prev_win = 'Start'
    for k, v in map_wins:
        row = {'Map': k, 'Result': v, 'PreviousResult': prev_win}
        row.update(base_row)
        data.append(row)
        prev_win = v
    if full_data:
        parse_timeline(id, map_wins)
    players.update(team1_players + team2_players)
    vod = get_vod_link(soup)
    match_metadata = {'event': event,
                      'event_id': event_id,
                      'team1': team1_name,
                      'team1_id': team1_id,
                      'team2': team2_name,
                      'team2_id': team2_id,
                      'team1_players': team1_players,
                      'team2_players': team2_players
                      }
    if vod is not None:
        match_metadata['vod'] = vod
    else:
        match_metadata['vod'] = ''
    with open(os.path.join(match_dir, 'metadata.json'), 'w', encoding='utf8') as f:
        json.dump(match_metadata, f, sort_keys=True, indent=4)
    for player_name, v in hero_picks.items():
        if player_name in team1_players:
            other_players = [x for x in team1_players if x != player_name]
            team = team1_name
        else:
            other_players = [x for x in team2_players if x != player_name]
            team = team2_name

        base_row = {'Date': str(date), 'Player': player_name, 'WinstonLabID': id, 'Team': team}
        for i in range(5):
            base_row['Teammate{}'.format(i + 1)] = other_players[i]
        for map_name, v2 in v.items():
            row = {'Map': map_name}
            row.update(base_row)
            row.update(v2)
            hero_pick_data.append(row)
    return data, hero_pick_data


def output_data(data, hero_pick_data):
    game_header = ['Date', 'Team1', 'Team2', 'Result',
                   'PreviousResult',
                   'Map',
                   'WinstonLabID', 'EventName', 'EventID', 'FullData',
                   'Team1_player1', 'Team1_player2', 'Team1_player3', 'Team1_player4', 'Team1_player5', 'Team1_player6',
                   'Team2_player1', 'Team2_player2', 'Team2_player3', 'Team2_player4', 'Team2_player5',
                   'Team2_player6', ]
    hero_header = ['Date', 'Player', 'Map', 'WinstonLabID', 'Team', 'Teammate1', 'Teammate2', 'Teammate3', 'Teammate4',
                   'Teammate5', 'Teammate6'] + sorted(heroes)
    with open(os.path.join('raw_data', 'matches.txt'), 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, game_header)
        writer.writeheader()
        for r in data:
            writer.writerow(r)

    with open(os.path.join('raw_data', 'hero_pick_rate.txt'), 'w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, hero_header)
        writer.writeheader()
        for r in hero_pick_data:
            for h in heroes:
                if h not in r:
                    r[h] = 0
            writer.writerow(r)

    with open(os.path.join('raw_data', 'heroes.txt'), 'w', encoding='utf8') as f:
        for h in sorted(heroes):
            f.write('{}\n'.format(h))

    with open(os.path.join('raw_data', 'players.txt'), 'w', encoding='utf8') as f:
        for h in sorted(players):
            f.write('{}\n'.format(h))

    with open(os.path.join('raw_data', 'maps.txt'), 'w', encoding='utf8') as f:
        for h in sorted(maps):
            f.write('{}\n'.format(h))

    with open(os.path.join('raw_data', 'teams.txt'), 'w', encoding='utf8') as f:
        for h in sorted(teams):
            f.write('{}\n'.format(h))


def get_hero_pick_rates(soup):
    global heroes
    # hero picks
    hero_picks = {}
    for line in soup.prettify().splitlines():
        if 'heroStatsArr = heroStatsArr' in line:
            line = line.strip()
            line = line.replace('heroStatsArr = heroStatsArr.concat(', '')
            line = line.replace(');teamcompsArr2 = false;   initialize_heroStats(', '')
            line, _ = line.split(');heroStatsArr2')
            data = json.loads(line)
            for x in data:
                player_name = x['playerName'].lower()
                map = x['map'].lower()
                if player_name not in hero_picks:
                    hero_picks[player_name] = {}
                if map not in hero_picks[player_name]:
                    hero_picks[player_name][map] = {}
                game_number = x['gameNumber']
                round_type = x['roundtype']
                hero = x['hero'].lower()
                heroes.add(hero)
                time_played = float(x['timePlayed'])
                hero_picks[player_name][map][hero] = time_played

    for player_name, v in hero_picks.items():
        for map_name, v2 in v.items():
            total = sum(v2.values())
            for hero, v3 in v2.items():
                hero_picks[player_name][map_name][hero] = v3 / total
    return hero_picks


def scrape_match_data():
    data, hero_pick_data = [], []
    errors = []
    not_found = []
    for i in range(1, 2500):
        if i not in test_files:
            continue
        # if i != 513:
        #    continue
        if i in [201, 685, 1658]: # Bad ones
            continue
        print(i)
        try:
            d, hpd = parse_page(i)
            parse_caps(i)
        except NotFoundID:
            not_found.append(str(i))
            continue
        except Exception as e:
            raise
            print(e)
            errors.append(str(i))
            continue
        if d is None:
            continue
        data.extend(d)
        hero_pick_data.extend(hpd)
        time.sleep(0.2)
    output_data(data, hero_pick_data)
    with open('errors.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(errors))
    with open('not_found.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(not_found))

def parse_caps(id):
    match_dir = os.path.join(annotations_dir, str(id))
    for gi in range(1, 7):
        begins, ends = [], []
        game_meta = None
        game_dir = os.path.join(match_dir, str(gi))
        if not os.path.exists(game_dir):
            continue
        for ri in range(1, 20):
            if os.path.exists(os.path.join(game_dir, '{}_{}_caps.txt'.format(gi, ri))):
                continue
            page = requests.get('https://www.winstonslab.com/matches/mapAndPtCaps.php?matchID={}&gameNumber={}&roundNumber={}'.format(id, gi, ri))
            try:
                resp = page.content.decode('utf8')
            except:
                continue
            if not resp:
                print(gi, ri)
                break
            data = json.loads(resp)
            with open(os.path.join(game_dir, '{}_{}_caps.txt'.format(gi, ri)), 'w', encoding='utf8',
                         newline='') as dataf:
                json.dump(data, dataf)


def parse_timeline(id, map_wins):
    maps = [x[0] for x in map_wins]
    print(map_wins)
    match_dir = os.path.join(annotations_dir, str(id))
    for gi in range(1, 7):
        begins, ends = [], []
        game_meta = None
        for ri in range(1, 20):

            page = requests.get(
                'https://www.winstonslab.com/matches/getMatchData.php?matchID={}&gameNumber={}&roundNumber={}'.format(
                    id, gi, ri))
            try:
                resp = page.content.decode('utf8')
            except:
                continue
            if not resp:
                print(gi, ri)
                break
            data = json.loads(resp)
            if ri == 1:
                game_dir = os.path.join(match_dir, str(gi))
                os.makedirs(game_dir, exist_ok=True)
                game_meta = {k: v for k, v in data.items() if k not in ['events', 'picks']}
                game_meta['map'] = maps[gi-1]
            begins.append(int(data['events'][0][0]))
            ends.append(int(data['events'][-1][0]))
            with open(os.path.join(game_dir, '{}_{}_picks.txt'.format(gi, ri)), 'w', encoding='utf8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ['time', 'player1', 'player2', 'player3', 'player4', 'player5', 'player6', 'player7', 'player8',
                     'player9', 'player10', 'player11', 'player12'])
                for line in data['picks']:
                    writer.writerow(line)
            with open(os.path.join(game_dir, '{}_{}_deaths.txt'.format(gi, ri)), 'w', encoding='utf8',
                      newline='') as deathf, \
                    open(os.path.join(game_dir, '{}_{}_kills.txt'.format(gi, ri)), 'w', encoding='utf8',
                         newline='') as killf, \
                    open(os.path.join(game_dir, '{}_{}_ult_gains.txt'.format(gi, ri)), 'w', encoding='utf8',
                         newline='') as ult_gainf, \
                    open(os.path.join(game_dir, '{}_{}_ult_uses.txt'.format(gi, ri)), 'w', encoding='utf8',
                         newline='') as ult_usef, \
                    open(os.path.join(game_dir, '{}_{}_switches.txt'.format(gi, ri)), 'w', encoding='utf8',
                         newline='') as switchf, \
                    open(os.path.join(game_dir, '{}_{}_data.txt'.format(gi, ri)), 'w', encoding='utf8',
                         newline='') as dataf, \
                    open(os.path.join(game_dir, '{}_{}_pauses.txt'.format(gi, ri)), 'w', encoding='utf8',
                         newline='') as pausef:
                types = ['DEATH', 'KILL', 'ULT_GAIN', 'ULT_USE', 'SWITCH']
                headers = {'DEATH': ['time', 'team', 'player_number', 'character'],
                           'KILL': ['time', 'team', 'player_number', 'character', 'ability', 'something'],
                           'ULT_GAIN': ['time', 'team', 'player_number', 'character'],
                           'ULT_USE': ['time', 'team', 'player_number', 'character'],
                           'SWITCH': ['time', 'team', 'player_number', 'from_character', 'to_character']}
                writers = {'DEATH': csv.writer(deathf),
                           'KILL': csv.writer(killf),
                           'ULT_GAIN': csv.writer(ult_gainf),
                           'ULT_USE': csv.writer(ult_usef),
                           'SWITCH': csv.writer(switchf), }
                for k, v in writers.items():
                    v.writerow(headers[k])
                json.dump(data, dataf)
                for line in data['events']:
                    for t in types:
                        if line[1] == t:
                            writers[t].writerow([x for x in line if x != t])
                            break
                    try:
                        if 'PAUSE' in line[1]:
                            pausef.write('{}\n'.format(','.join(map(str, line))))
                    except:
                        continue
        if game_meta is None:
            continue
        game_meta['begins'] = begins
        game_meta['ends'] = ends
        with open(os.path.join(game_dir, 'meta.json'), 'w', encoding='utf8') as f:
            json.dump(game_meta, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    scrape_match_data()
    # scrape_timeline_data()
