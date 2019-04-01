import os
import requests
import csv
import time

directory = os.path.dirname(os.path.abspath(__file__))
resp = requests.get('https://api.overwatchleague.com/stats/players')


with open(os.path.join(directory, 'stats.csv'), 'w', newline='', encoding='utf8') as f:
    headers = ['role', 'name', 'team', 'eliminations_avg_per_10m',
                                'deaths_avg_per_10m',  'hero_damage_avg_per_10m',
                                'healing_avg_per_10m', 'time_played_total']
    writer = csv.DictWriter(f, headers,)
    for d in resp.json()['data']:
        new_d = {k: d[k] for k in headers}
        m, s = divmod(new_d['time_played_total'], 60)
        h, m = divmod(m, 60)
        new_d['time_played_total'] = '{}h {}m'.format(int(h), int(m))
        writer.writerow(new_d)
