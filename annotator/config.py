
local_directory = r'E:\Data\Overwatch\raw_data\annotations\matches'
vod_directory = r'E:\Data\Overwatch\raw_data\vods'
training_data_directory = r'E:\Data\Overwatch\training_data'

DEV = False

if DEV:
    print()
    print('WARNING: Using the local dev server!')
    print()
    print()
    site_url = 'http://localhost:8000/'
else:
    site_url = 'https://omnicintelligence.com/'

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

MID_HEIGHT = 96
MID_WIDTH = 288

REPLAY_HEIGHT = 64
REPLAY_WIDTH = 208

PAUSE_HEIGHT = 48
PAUSE_WIDTH = 144

SMALLER_WINDOW_WIDTH = 128
SMALLER_WINDOW_HEIGHT = 128

FRAME_HEIGHT = 720
FRAME_WIDTH = 1280

BOX_PARAMETERS = {
    'O': {
        'WINDOW': {
            'X': 0,
            'Y': 0,
            'HEIGHT': FRAME_HEIGHT,
            'WIDTH': FRAME_WIDTH,
        },
        'MID': {
            'X': 496,
            'Y': 37,
            'HEIGHT': MID_HEIGHT,
            'WIDTH': MID_WIDTH,
        },
        'GAME': {
            'X': 520,
            'Y': 35,
            'HEIGHT': 84,
            'WIDTH': 240,
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
            'WIDTH': REPLAY_WIDTH,
            'HEIGHT': REPLAY_HEIGHT,
        },
        'PAUSE': {
            'X': 550,
            'Y': 310,
            'WIDTH': PAUSE_WIDTH,
            'HEIGHT': PAUSE_HEIGHT,
        }
    }
}

BOX_PARAMETERS['KO'] = {
        'WINDOW': {
            'X': 0,
            'Y': 0,
            'HEIGHT': FRAME_HEIGHT,
            'WIDTH': FRAME_WIDTH,
        },
    'MID': {
        'X': 496,
        'Y': 45,
        'HEIGHT': MID_HEIGHT,
        'WIDTH': MID_WIDTH,
    },
    'GAME': {
        'X': 520,
        'Y': 43,
        'HEIGHT': 84,
        'WIDTH': 240,
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
        'WIDTH': REPLAY_WIDTH,
        'HEIGHT': REPLAY_HEIGHT,
    },
    'PAUSE': {
        'X': 550,
        'Y': 310 + 13,
        'WIDTH': PAUSE_WIDTH,
        'HEIGHT': PAUSE_HEIGHT,
    }
}

BOX_PARAMETERS['W'] = {
        'WINDOW': {
            'X': 0,
            'Y': 0,
            'HEIGHT': FRAME_HEIGHT,
            'WIDTH': FRAME_WIDTH,
        },
    'MID': {
        'X': 496,
        'Y': 36,
        'HEIGHT': MID_HEIGHT,
        'WIDTH': MID_WIDTH,
    },
    'GAME': {
        'X': 520,
        'Y': 34,
        'HEIGHT': 84,
        'WIDTH': 240,
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
        'WIDTH': REPLAY_WIDTH,
        'HEIGHT': REPLAY_HEIGHT,
    },
    'PAUSE': {
        'X': 550,
        'Y': 300,
        'WIDTH': PAUSE_WIDTH,
        'HEIGHT': PAUSE_HEIGHT,
    }
}

BOX_PARAMETERS['K'] = {
        'WINDOW': {
            'X': 0,
            'Y': 0,
            'HEIGHT': FRAME_HEIGHT,
            'WIDTH': FRAME_WIDTH,
        },
    'MID': {
        'X': 496,
        'Y': 62,
        'HEIGHT': MID_HEIGHT,
        'WIDTH': MID_WIDTH,
    },
    'GAME': {
        'X': 520,
        'Y': 60,
        'HEIGHT': 84,
        'WIDTH': 240,
},
    'KILL_FEED': {
        'X': 1280 - 20 - 248,
        'Y': 145,
        'WIDTH': 248,
        'HEIGHT': 256
    },
    'KILL_FEED_SLOT': {
        'X': 1280 - 20 - 248,
        'Y': 145,
        'WIDTH': 248,
        'HEIGHT': 32,
        'MARGIN': 4
    },
    'LEFT': {
        'X': 31,
        'Y': 77,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 7,
    },
    'RIGHT': {
        'X': 834,
        'Y': 77,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 7,
    },
}

BOX_PARAMETERS['2'] = {
        'WINDOW': {
            'X': 0,
            'Y': 0,
            'HEIGHT': FRAME_HEIGHT,
            'WIDTH': FRAME_WIDTH,
        },
    'MID': {
        'X': 496,
        'Y': 62,
        'HEIGHT': MID_HEIGHT,
        'WIDTH': MID_WIDTH,
    },
    'GAME': {
        'X': 520,
        'Y': 60,
        'HEIGHT': 84,
        'WIDTH': 240,
},
    'KILL_FEED': {
        'X': 1280 - 20 - 248,
        'Y': 140,
        'WIDTH': 248,
        'HEIGHT': 256
    },
    'KILL_FEED_SLOT': {
        'X': 1280 - 20 - 248,
        'Y': 140,
        'WIDTH': 248,
        'HEIGHT': 32,
        'MARGIN': 4
    },
    'LEFT': {
        'X': 31,
        'Y': 72,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 7,
    },
    'RIGHT': {
        'X': 827,
        'Y': 72,
        'WIDTH': 64,
        'HEIGHT': 64,
        'MARGIN': 7,
    },
    'ZOOMED_LEFT': {
        'X': 36,
        'Y': 72,
        'WIDTH': 80,
        'HEIGHT': 80,
        'MARGIN': 7,
    },
    'ZOOMED_RIGHT': {
        'X': 738,
        'Y': 72,
        'WIDTH': 80,
        'HEIGHT': 80,
        'MARGIN': 7,
    },
    'REPLAY': {
        'X': 145,
        'Y': 150,
        'WIDTH': REPLAY_WIDTH,
        'HEIGHT': REPLAY_HEIGHT,
    },
    'PAUSE': {
        'X': 530,
        'Y': 300,
        'WIDTH': PAUSE_WIDTH,
        'HEIGHT': PAUSE_HEIGHT,
    },
    'SMALLER_WINDOW':{
        'X': 1280 -128 - 10,
        'Y': 128 - 10,
        'WIDTH': SMALLER_WINDOW_WIDTH,
        'HEIGHT': SMALLER_WINDOW_HEIGHT,
    }
}

BOX_PARAMETERS['A'] = {  # Black borders around video feed
        'WINDOW': {
            'X': 0,
            'Y': 0,
            'HEIGHT': FRAME_HEIGHT,
            'WIDTH': FRAME_WIDTH,
        },
    'MID': {
        'X': 491,
        'Y': 47,
        'HEIGHT': MID_HEIGHT,
        'WIDTH': MID_WIDTH,
    },
    'GAME': {
        'X': 490,
        'Y': 45,
        'HEIGHT': 84,
        'WIDTH': 230,
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
        'WIDTH': REPLAY_WIDTH,
        'HEIGHT': REPLAY_HEIGHT,
    },
    'PAUSE': {
        'X': 550,
        'Y': 300,
        'WIDTH': PAUSE_WIDTH,
        'HEIGHT': PAUSE_HEIGHT,
    }
}

for k, f in BOX_PARAMETERS.items():
    for s in ['LEFT', 'RIGHT']:
        BOX_PARAMETERS[k]['{}_NAME'.format(s)] = {'X': f[s]['X'],
                                                  'Y': f[s]['Y'] + 32,
                                                  'WIDTH': f[s]['WIDTH'],
                                                  'HEIGHT': 16,
                                                  'MARGIN': f[s]['MARGIN'],
                                                  }

