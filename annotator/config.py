
local_directory = r'E:\Data\Overwatch\raw_data\annotations\matches'
vod_directory = r'E:\Data\Overwatch\raw_data\vods'
training_data_directory = r'E:\Data\Overwatch\training_data'

site_url = 'http://localhost:8000/'
#site_url = 'https://omnicintelligence.com/'

api_url = site_url + 'api/'

na_lab = 'n/a'
sides = ['left', 'right']



PLATE_ASSIST_WIDTH = 16
PLATE_ASSIST_MARGIN = 3
PLATE_LEFT_MARGIN = 3
PLATE_NAME_LEFT_MARGIN = 6
PLATE_RIGHT_MARGIN = 9
PLATE_PORTRAIT_WIDTH = 36
PRIMARY_ABILITY_WIDTH = 25
SPECIAL_ABILITY_WIDTH = 50

CHAR_VALUES = {'a': 6,
               'b': 5,
               'c': 6,
               'd': 7,
               'e': 4,
               'f': 4,
               'g': 8,
               'h': 7,
               'i': 2,
               'j': 4,
               'k': 6,
               'l': 4,
               'm': 8,
               'n': 7,
               'o': 8,
               'p': 5,
               'q': 8,
               'r': 5,
               's': 5,
               't': 4,
               'u': 7,
               'v': 6,
               'w': 9,
               'x': 7,
               'y': 6,
               'z': 6,
               'default': 6
               }

stage_2_shift = 25

## Player boundary guidelines

# Aligned to nameplate, left is top left corner of nameplate

# Top margin = Bottom margin

PLAYER_WIDTH = 64
PLAYER_HEIGHT = 64
KILL_FEED_WIDTH = 248
KILL_FEED_HEIGHT = 32

BASE_PLAYER_Y = 41
BASE_LEFT_X = 29
BASE_RIGHT_X = 832
BASE_PLAYER_MARGIN = 7

MID_HEIGHT = 100
MID_WIDTH = 290

BOX_PARAMETERS = {
    'O': {
        'MID': {
            'X': 495,
            'Y': 35,
            'HEIGHT': MID_HEIGHT,
            'WIDTH': MID_WIDTH,
        },
        'KILL_FEED': {
            'X': 1280 - 20 - 250,
            'Y': 112,
            'WIDTH': 256,
            'HEIGHT': 256
        },
        'KILL_FEED_SLOT': {
            'X': 1280 - 20 - 248,
            'Y': 112,
            'WIDTH': 248,
            'HEIGHT': 32,
            'MARGIN': 3
        },
        'LEFT': {
            'X': BASE_LEFT_X,
            'Y': BASE_PLAYER_Y,
            'WIDTH': PLAYER_WIDTH,
            'HEIGHT': PLAYER_HEIGHT,
            'MARGIN': BASE_PLAYER_MARGIN,
        },
        'RIGHT': {
            'X': BASE_RIGHT_X,
            'Y': BASE_PLAYER_Y,
            'WIDTH': PLAYER_WIDTH,
            'HEIGHT': PLAYER_HEIGHT,
            'MARGIN': BASE_PLAYER_MARGIN,
        },
        'REPLAY': {
            'X': 105,
            'Y': 110,
            'WIDTH': 210,
            'HEIGHT': 60,
        },
        'PAUSE': {
            'X': 550,
            'Y': 310,
            'WIDTH': 150,
            'HEIGHT': 40,
        }
    }
}

BOX_PARAMETERS['K'] = {
    'MID': {
        'X': 495,
        'Y': 43,
        'HEIGHT': MID_HEIGHT,
        'WIDTH': MID_WIDTH,
    },
    'KILL_FEED': {
        'X': 1280 - 20 - 248 - 20,
        'Y': 120,
        'WIDTH': 248,
        'HEIGHT': 256
    },
    'KILL_FEED_SLOT': {
        'X': 1280 - 20 - 248 - 20,
        'Y': 120,
        'WIDTH': 248,
        'HEIGHT': 32,
        'MARGIN': 1
    },
    'LEFT': {
        'X': 46,
        'Y': 49,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': 5,
    },
    'RIGHT': {
        'X': 827,
        'Y': 49,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': 4,
    },
    'REPLAY': {
        'X': 105,
        'Y': 110 + 13,
        'WIDTH': 210,
        'HEIGHT': 60,
    },
    'PAUSE': {
        'X': 550,
        'Y': 310 + 13,
        'WIDTH': 150,
        'HEIGHT': 40,
    }
}

BOX_PARAMETERS['W'] = {
    'MID': {
        'X': 495,
        'Y': 34,
        'HEIGHT': MID_HEIGHT,
        'WIDTH': MID_WIDTH,
    },
    'KILL_FEED': {
        'X': 1280 - 20 - 248,
        'Y': 112,
        'WIDTH': 256,
        'HEIGHT': 256
    },
    'KILL_FEED_SLOT': {
        'X': 1280 - 20 - 248,
        'Y': 112,
        'WIDTH': 248,
        'HEIGHT': 32,
        'MARGIN': 2
    },
    'LEFT': {
        'X': 29,
        'Y': 41,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 7,
    },
    'RIGHT': {
        'X': 832,
        'Y': 41,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 7,
    },
    'REPLAY': {
        'X': 125,
        'Y': 165,
        'WIDTH': 210,
        'HEIGHT': 60,
    },
    'PAUSE': {
        'X': 550,
        'Y': 300,
        'WIDTH': 190,
        'HEIGHT': 70,
    }
}

BOX_PARAMETERS['2'] = {
    'MID': {
        'X': 495,
        'Y': 60,
        'HEIGHT': MID_HEIGHT,
        'WIDTH': MID_WIDTH,
    },
    'KILL_FEED': {
        'X': 1280 - 20 - 248,
        'Y': 136,
        'WIDTH': 248,
        'HEIGHT': 256
    },
    'KILL_FEED_SLOT': {
        'X': 1280 - 20 - 248,
        'Y': 136,
        'WIDTH': 248,
        'HEIGHT': 32,
        'MARGIN': 2
    },
    'LEFT': {
        'X': 31,
        'Y': 67,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 7,
    },
    'RIGHT': {
        'X': 827,
        'Y': 67,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 7,
    },
    'REPLAY': {
        'X': 145,
        'Y': 150,
        'WIDTH': 210,
        'HEIGHT': 60,
    },
    'PAUSE': {
        'X': 530,
        'Y': 300,
        'WIDTH': 210,
        'HEIGHT': 80,
    }
}

BOX_PARAMETERS['A'] = {  # Black borders around video feed
    'MID': {
        'X': 490,
        'Y': 45,
        'HEIGHT': MID_HEIGHT,
        'WIDTH': MID_WIDTH,
    },
    'KILL_FEED': {
        'X': 950,
        'Y': 115,
        'WIDTH': 270,
        'HEIGHT': 205
    },
    'LEFT': {
        'X': 51,
        'Y': 45,
        'WIDTH': 67,
        'HEIGHT': 55,
        'MARGIN': 1,
    },
    'RIGHT': {
        'X': 825,
        'Y': 45,
        'WIDTH': 67,
        'HEIGHT': 55,
        'MARGIN': 1,
    },
    'REPLAY': {
        'X': 115,
        'Y': 120,
        'WIDTH': 150,
        'HEIGHT': 40,
    },
    'PAUSE': {
        'X': 550,
        'Y': 300,
        'WIDTH': 190,
        'HEIGHT': 70,
    }
}

for k, f in BOX_PARAMETERS.items():
    for s in ['LEFT', 'RIGHT']:
        BOX_PARAMETERS[k]['{}_NAME'.format(s)] = {'X': f[s]['X'],
                                                  'Y': f[s]['Y'] + 34,
                                                  'WIDTH': f[s]['WIDTH'],
                                                  'HEIGHT': 12,
                                                  'MARGIN': f[s]['MARGIN'],
                                                  }

