
BASE_TIME_STEP = 0.3

local_directory = r'N:\Data\Overwatch\raw_data\annotations\matches'
vod_directory = r'N:\Data\Overwatch\raw_data\vods'
training_data_directory = r'N:\Data\Overwatch\training_data'

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


offsets = {3120: 8.6,
           3117: 0.1,
           3118: 8.1,
           3119: 3.9,
           3121: 1.7,
           3122: 5.1,
           3123: 6.3,
           3124: 3.1,
           3125: 8.3,
           3127: 7.5,
           3128: 2.6,
           3129: 3.8,
           3130: 6.3,
           3134: 5.3,
           3135: 4.4,
           3136: 6.3,
           3137: 7,
           3138: 7.6,
           3139: 1.6,
           3140: 3,
           3141: 7.3,
           3144: 5.8,
           3145: 2.3,
           3146: 4.1,
           3147: 3.1,
           3148: 9.3,
           3149: 2.4,
           3150: 0.1,
           3151: 1.9,
           3152: 5.1,
           3158: 9.4,
           3159: 3.9,
           3161: 6.3,
           3162: 2.3,
           3163: 6.2,
           #3639: -5.7,
           3786: 2.6,
           2701: 0.1,
           2808: 0.1,
           3958: 0.1,
           3956: 0.1,
           3955: 0.1,
           3954: 0.1,
           3953: 0.1,
           #3969: 0.1,
           }



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

PLAYER_WIDTH = 72
PLAYER_HEIGHT = 72
PLAYER_NAME_WIDTH = 80
KILL_FEED_WIDTH = 296
KILL_FEED_HEIGHT = 32
KILL_FEED_OFFSET = 40

BASE_PLAYER_Y = 37
BASE_LEFT_X = 29
BASE_RIGHT_X = 832
BASE_PLAYER_MARGIN = -1

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
        'KILL_FEED_SLOT': {
            'X': 1280 - KILL_FEED_OFFSET - KILL_FEED_WIDTH,
            'Y': 112,
            'WIDTH': KILL_FEED_WIDTH,
            'HEIGHT': 32,
            'MARGIN': 3
        },
        'LEFT': {
            'X': BASE_LEFT_X,
            'Y': BASE_PLAYER_Y + 1,
            'WIDTH': PLAYER_WIDTH,
            'HEIGHT': PLAYER_HEIGHT,
            'MARGIN': BASE_PLAYER_MARGIN,
        },
        'RIGHT': {
            'X': BASE_RIGHT_X,
            'Y': BASE_PLAYER_Y + 1,
            'WIDTH': PLAYER_WIDTH,
            'HEIGHT': PLAYER_HEIGHT,
            'MARGIN': BASE_PLAYER_MARGIN,
        },
    }
}

BOX_PARAMETERS['1'] = {
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
    'KILL_FEED_SLOT': {
        'X': 1280 - KILL_FEED_OFFSET - KILL_FEED_WIDTH - 20,
        'Y': 120,
        'WIDTH': KILL_FEED_WIDTH,
        'HEIGHT': 32,
        'MARGIN': 1
    },
    'LEFT': {
        'X': 46,
        'Y': 46,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': -3,
    },
    'RIGHT': {
        'X': 827,
        'Y': 46,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': -4,
    }
}

BOX_PARAMETERS['U'] = {
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
        'KILL_FEED_SLOT': {
            'X': 1280 - KILL_FEED_OFFSET - KILL_FEED_WIDTH,
            'Y': 116,
            'WIDTH': KILL_FEED_WIDTH,
            'HEIGHT': 32,
            'MARGIN': 3
        },
        'LEFT': {
            'X': BASE_LEFT_X,
            'Y': BASE_PLAYER_Y + 7,
            'WIDTH': PLAYER_WIDTH,
            'HEIGHT': PLAYER_HEIGHT,
            'MARGIN': BASE_PLAYER_MARGIN,
        },
        'RIGHT': {
            'X': BASE_RIGHT_X,
            'Y': BASE_PLAYER_Y + 7,
            'WIDTH': PLAYER_WIDTH,
            'HEIGHT': PLAYER_HEIGHT,
            'MARGIN': BASE_PLAYER_MARGIN,
        },
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
    'KILL_FEED_SLOT': {
        'X': 1280 - KILL_FEED_OFFSET - KILL_FEED_WIDTH,
        'Y': 112,
        'WIDTH': KILL_FEED_WIDTH,
        'HEIGHT': 32,
        'MARGIN': 2
    },
    'LEFT': {
        'X': 29,
        'Y': BASE_PLAYER_Y,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': BASE_PLAYER_MARGIN,
    },
    'RIGHT': {
        'X': 832,
        'Y': BASE_PLAYER_Y,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': BASE_PLAYER_MARGIN,
    },
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
    'KILL_FEED_SLOT': {
        'X': 1280 - KILL_FEED_OFFSET - KILL_FEED_WIDTH,
        'Y': 146,
        'WIDTH': KILL_FEED_WIDTH,
        'HEIGHT': 32,
        'MARGIN': 3
    },
    'LEFT': {
        'X': 34,
        'Y': 72,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': BASE_PLAYER_MARGIN -1,
    },
    'RIGHT': {
        'X': 835,
        'Y': 72,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': BASE_PLAYER_MARGIN - 1,
    },
}

BOX_PARAMETERS['G'] = {
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
    'KILL_FEED_SLOT': {
        'X': 1280 - KILL_FEED_OFFSET - KILL_FEED_WIDTH,
        'Y': 143,
        'WIDTH': KILL_FEED_WIDTH,
        'HEIGHT': 32,
        'MARGIN': 2
    },
    'LEFT': {
        'X': 34,
        'Y': 68,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': BASE_PLAYER_MARGIN,
    },
    'RIGHT': {
        'X': 834,
        'Y': 68,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': BASE_PLAYER_MARGIN - 1,
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
    'KILL_FEED_SLOT': {
        'X': 1280 - KILL_FEED_OFFSET - KILL_FEED_WIDTH,
        'Y': 140,
        'WIDTH': KILL_FEED_WIDTH,
        'HEIGHT': 32,
        'MARGIN': 3
    },
    'LEFT': {
        'X': 33,
        'Y': 67,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': -1,
    },
    'RIGHT': {
        'X': 827,
        'Y': 67,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': -1,
    },
    'ZOOMED_LEFT': {
        'X': 36,
        'Y': 67,
        'WIDTH': 88,
        'HEIGHT': 88,
        'MARGIN': -1,
    },
    'ZOOMED_RIGHT': {
        'X': 738,
        'Y': 67,
        'WIDTH': 88,
        'HEIGHT': 88,
        'MARGIN': -1,
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
    'KILL_FEED': {
        'X': 950,
        'Y': 115,
        'WIDTH': 270,
        'HEIGHT': 205
    },
    'LEFT': {
        'X': 51,
        'Y': 41,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': BASE_PLAYER_MARGIN,
    },
    'RIGHT': {
        'X': 825,
        'Y': 41,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': BASE_PLAYER_MARGIN,
    },
}

BOX_PARAMETERS['3'] = {
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
    'KILL_FEED_SLOT': {
        'X': 1280 - KILL_FEED_OFFSET - KILL_FEED_WIDTH,
        'Y': 140,
        'WIDTH': KILL_FEED_WIDTH,
        'HEIGHT': 32,
        'MARGIN': 3
    },
    'LEFT': {
        'X': 33,
        'Y': 67,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': -1,
    },
    'RIGHT': {
        'X': 827,
        'Y': 67,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': -1,
    },
    'ZOOMED_LEFT': {
        'X': 22,
        'Y': 65,
        'WIDTH': 92,
        'HEIGHT': 92,
        'MARGIN': 3,
    },
    'ZOOMED_RIGHT': {
        'X': 700,
        'Y': 65,
        'WIDTH': 92,
        'HEIGHT': 92,
        'MARGIN': 3,
    }
}

BOX_PARAMETERS['32'] = {
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
    'KILL_FEED_SLOT': {
        'X': 1280 - KILL_FEED_OFFSET - KILL_FEED_WIDTH,
        'Y': 140,
        'WIDTH': KILL_FEED_WIDTH,
        'HEIGHT': 32,
        'MARGIN': 3
    },
    'LEFT': {
        'X': 33,
        'Y': 67,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': -1,
    },
    'RIGHT': {
        'X': 827,
        'Y': 67,
        'WIDTH': PLAYER_WIDTH,
        'HEIGHT': PLAYER_HEIGHT,
        'MARGIN': -1,
    },
    'ZOOMED_LEFT': {
        'X': 29,
        'Y': 63,
        'WIDTH': 84,
        'HEIGHT': 84,
        'MARGIN': 1,
    },
    'ZOOMED_RIGHT': {
        'X': 755,
        'Y': 63,
        'WIDTH': 84,
        'HEIGHT': 84,
        'MARGIN': 1,
    }
}


for k, f in BOX_PARAMETERS.items():
    for s in ['LEFT', 'RIGHT']:
        diff = PLAYER_NAME_WIDTH - PLAYER_WIDTH
        BOX_PARAMETERS[k]['{}_NAME'.format(s)] = {'X': f[s]['X'] - int(diff/2),
                                                  'Y': f[s]['Y'] + int(PLAYER_HEIGHT /2),
                                                  'WIDTH': PLAYER_NAME_WIDTH,
                                                  'HEIGHT': 16,
                                                  'MARGIN': f[s]['MARGIN'] - diff,
                                                  }

