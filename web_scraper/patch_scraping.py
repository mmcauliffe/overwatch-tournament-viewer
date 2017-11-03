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

patch_link_template = 'http://overwatch.wikia.com/wiki/{}'

page = requests.get('http://overwatch.wikia.com/wiki/List_of_Patches')

soup = BeautifulSoup(page.content, 'html.parser')

major_patch_candidates = soup.find_all('li')

patches = []

ignored_patches = ['1.15.0.2a', '1.13.0.2d', '1.13.0.2c', '1.13.0.2b', '1.12.0.2b', '1.9.0.2b', '1.7.0.2c', '1.7.0.2b',
                   '1.6.2.1', '1.6.1.2.33651', '1.6.0.2', '1.5.0.2.33207', '1.6.0.2', '1.4.0.2c', '1.4.0.2b',
                   '1.3.0.3b', '1.3.0.3a', '1.3.0.3', '1.2.0.3', '1.0.5.1', '1.0.4.2b']

for p in major_patch_candidates:
    if '(PS4/XB1 only)' in p.get_text():
        continue
    link = p.find_all('a')
    if not link:
        continue
    link = link[0]
    date_text = link.get_text()
    m = re.match(r'(\w+) (\d+), (\d+)', date_text)
    if m is None:
        continue
    print(date_text)
    id = link['title']
    month_name, day, year = m.groups()
    date = datetime.date(year=int(year), day=int(day), month=list(calendar.month_abbr).index(month_name[:3]))
    p = requests.get(patch_link_template.format(id))
    s = BeautifulSoup(p.content, 'html.parser')
    number_candidates = s.find_all('i')
    patch_name = None
    for n in number_candidates:
        m = re.match(r'(patch )?([0-9.abcd]{2,})( [(]pc only[)])?', n.get_text().lower())
        if m is None:
            continue
        print(m.groups())
        patch_name = m.groups()[1]
        break
    if patch_name is None:
        patch_name = 'Beta_{}_{}_{}'.format(date.year, date.month, date.day)
    if patch_name in ignored_patches:
        continue
    patches.append([patch_name, date])

header = ['PatchName', 'Begin', 'End', 'Length']
with open(os.path.join('raw_data', 'patches.txt'), 'w', encoding='utf8', newline='') as f:
    writer = csv.DictWriter(f, header)
    writer.writeheader()
    for i, p in enumerate(patches):
        row = {'PatchName': p[0], 'Begin': p[1]}
        if i == 0:
            row['End'] = datetime.date.today()
        else:
            row['End'] = patches[i - 1][1]
        row['Length'] = (row['End'] - row['Begin']).days
        writer.writerow(row)
