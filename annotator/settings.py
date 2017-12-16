import os

data_dir = r'E:\Data\Overwatch\raw_data'
annotations_dir = os.path.join(data_dir, 'annotations')

unusable = ['1633']

actual_starts = {
    '513': 1274,
    '1852': 880,
    '1853': 628,
    '1486': 464,
    '1492': 610,
    '1480': 413,
    '1881': 301,
    '1859': 649,
    '1858': 805,
    '1348': 1895
}

ignored = ['1812', '1815', '1817', '1819', '1633']
test_files = ['513', '1633', '1881', '1859', '1852', '1853', '1858', '1480', '2360', '2359', '2358', '2357', '2356',
              '2355', '1812', '1815', '1817', '1819']

heroes = ['doomfist', 'genji', 'mccree', 'pharah', 'reaper', 'soldier76', 'sombra', 'tracer',
          'bastion', 'hanzo', 'junkrat', 'mei', 'torbjorn', 'widowmaker',
          'd.va', 'orisa', 'reinhardt', 'roadhog', 'winston', 'zarya',
          'ana', 'lucio', 'mercy', 'moira', 'symmetra', 'zenyatta']

npcs = ['mech', 'turret', 'supercharger', 'teleporter', 'shieldgenerator', 'riptire']

damaging_abilities = {
    'doomfist': ['regular', 'headshot', 'rocketpunch', 'risinguppercut', 'seismicslam', 'meteorstrike'],
    'genji': ['regular', 'headshot', 'swiftstrike', 'dragonblade', 'deflect'],
    'mccree': ['regular', 'headshot', 'flashbang', 'deadeye'],
    'pharah': ['regular', 'barrage', 'concussiveblast'],
    'reaper': ['regular', 'headshot', 'deathblossom'],
    'soldier76': ['regular', 'headshot', 'tacticalvisor', 'helixrockets'],
    'sombra': ['regular', 'headshot'],
    'tracer': ['regular', 'headshot', 'pulsebomb'],
    'bastion': ['regular', 'headshot', 'configurationtank'],
    'hanzo': ['regular', 'headshot', 'scattershot', 'sonicarrow', 'dragonstrike'],
    'junkrat': ['regular', 'concussionmine', 'steeltrap', 'martyrdom', 'riptire'],
    'mei': ['regular', 'headshot', 'blizzard'],
    'torbjorn': ['regular', 'headshot', 'turret', 'hammer'],
    'widowmaker': ['regular', 'headshot', 'venommine'],
    'd.va': ['regular', 'headshot', 'selfdestruct', 'boosters', 'micromissiles'],
    'orisa': ['regular', 'headshot', 'halt'],
    'reinhardt': ['regular', 'firestrike', 'charge', 'earthshatter'],
    'roadhog': ['regular', 'headshot', 'wholehog', 'chainhook'],
    'winston': ['regular', 'primalrage', 'jumppack'],
    'zarya': ['regular', 'gravitonsurge'],
    'ana': ['regular', 'bioticgrenade', 'sleepdart'],
    'lucio': ['regular', 'headshot', 'soundwave'],
    'mercy': ['regular', 'headshot'],
    'moira': ['regular', 'bioticorb', 'coalescence'],
    'symmetra': ['regular', 'energyball', 'sentryturret'],
    'zenyatta': ['regular', 'headshot']
    }

revive_abilities = {'doomfist': [],
                    'genji': [],
                    'mccree': [],
                    'pharah': [],
                    'reaper': [],
                    'soldier76': [],
                    'sombra': [],
                    'tracer': [],
                    'bastion': [],
                    'hanzo': [],
                    'junkrat': [],
                    'mei': [],
                    'torbjorn': [],
                    'widowmaker': [],
                    'd.va': [],
                    'orisa': [],
                    'reinhardt': [],
                    'roadhog': [],
                    'winston': [],
                    'zarya': [],
                    'ana': [],
                    'lucio': [],
                    'mercy': ['resurrect'],
                    'moira': [],
                    'symmetra': [],
                    'zenyatta': []}

team_colors = ['red', 'blue']

maps = ['hanamura', 'temple of anubis', 'volskaya industries',
        'dorado', 'route 66', 'watchpoint: gibraltar',
        'ilios', 'lijiang tower', 'nepal', 'oasis',
        'eichenwalde', 'hollywood', "king's row", 'numbani']

player_box_y = {'regular': 45}
player_box_y['apex'] = player_box_y['regular']
left_box_x = {'regular': 30}
left_box_x['apex'] = left_box_x['regular'] + 21
right_box_x = {'regular': 830}
right_box_x['apex'] = right_box_x['regular'] - 5
player_box_height = 55
player_box_width = 67
left_box_width = 420
margin = {'regular':4}
margin['apex'] = margin['regular'] - 3


mid_box_y = player_box_y['regular']
mid_box_x = 490
mid_box_width = 300
mid_box_height = 140

kill_feed_box_y = 115
kill_feed_box_x = 950
kill_feed_box_width = 300
kill_feed_box_total_height = 205
kill_feed_box_height = int(kill_feed_box_total_height / 6)


def box(frame, team, number, style='regular'):
    if team == 'left':
        x = left_box_x[style]
    else:
        x = right_box_x[style]
    x += (player_box_width + margin[style]) * (number)
    return frame[player_box_y[style]: player_box_y[style] + player_box_height, x: x + player_box_width]


def kf_box(frame, number):
    y = kill_feed_box_y
    if number != 1:
        y += (kill_feed_box_height) * (number - 1)
    return frame[y: y + kill_feed_box_height, kill_feed_box_x: kill_feed_box_x + kill_feed_box_width]