import cv2
from annotator.utils import BOX_PARAMETERS, get_train_rounds, get_local_file, get_local_path, get_event_ranges, \
    get_kf_events, FileVideoStreamRange, Empty, construct_kf_at_time, calculate_hero_boundaries, calculate_ability_boundaries, calculate_first_hero_boundaries, CHAR_VALUES


def extract_kf_frames(rounds):
    params = BOX_PARAMETERS['REGULAR']['KILL_FEED_SLOT']
    for round_index, r in enumerate(rounds):
        print(round_index, len(rounds))
        print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'], r['id'])
        events = get_kf_events(r['id'])
        ranges = get_event_ranges(events, r['end'])

        fvs = FileVideoStreamRange(get_local_path(r), r['begin'], ranges, time_step=2).start()
        while True:
            try:
                frame, time_point = fvs.read()
            except Empty:
                break
            x = params['X']
            y = params['Y']
            # y += (params['HEIGHT'] + params['MARGIN']) * 0
            shape = frame.shape
            margin = 20
            box = frame[y: y + params['HEIGHT'],
                  x: x + params['WIDTH']]
            kf = construct_kf_at_time(events, time_point)
            width = 250
            right_x = shape[1] - margin
            for i,e in enumerate(kf):
                y = params['Y']
                y += (params['HEIGHT'] + params['MARGIN']) * i
                print(i, e)
                if e['second_player'] == 'n/a':
                    continue
                portrait_left, portrait_right = calculate_hero_boundaries(e['second_player'])
                ability_left, ability_right = calculate_ability_boundaries(portrait_left, e['ability'])
                first_hero_left, first_hero_right = calculate_first_hero_boundaries(ability_left, len(e['assisting_heroes']))
                cv2.imshow('second_hero', frame[y:y + params['HEIGHT'], right_x-portrait_left:right_x-portrait_right])
                cv2.imshow('ability', frame[y:y + params['HEIGHT'], right_x-ability_left:right_x-ability_right])
                cv2.imshow('first_hero', frame[y:y + params['HEIGHT'], right_x-first_hero_left:right_x-first_hero_right])

                cv2.imshow('frame', frame[y:y + params['HEIGHT'], right_x-width:right_x])
                cv2.waitKey(0)


def main():
    rounds = get_train_rounds()
    for r in rounds:
        print(r['sequences'])
        local_path = get_local_path(r)
        if local_path is None:
            print(r['game']['match']['wl_id'], r['game']['game_number'], r['round_number'])
            get_local_file(r)
    extract_kf_frames(rounds)


expected = {'cocco': 38,
            'cwoosh': 49,
            'mickie': 34,
            'effect': 33,
            'logix': 31,
            'custa': 31,
            'harryhook': 67,
            'tviq': 23,
            'zebbosai': 49,
            'zuppeh': 38,
            'chipshajen': 63,
            'manneten': 58,
            'mano': 34,
            'meko': 29,
            'libero': 33,
            'ark': 20,
            'saebyeolbe': 61,
            'fleta': 26,
            'pine': 21,
            'kuki': 22,
            'ryujehong': 63,
            'wekeed': 40,
            'zunba': 35,
            'tobi': 21,
            'seagull': 43,
            'xqc': 22,
            'janus': 33,
            'jjonak': 41,
            'neko': 28,
            'gamsu': 37,
            'dreamkazper': 76,
            'striker': 37,
            'kalios': 36,
            'kellex': 35,
            'taimou': 40,
            'silkthread': 59,
            'unkoe': 36,
            'soon': 30,
            'fate': 21,
            'envy': 26,
            'kariv': 30,
            'shaz': 26,
            'surefour': 51,
            'rascal': 38,
            'iremiix': 37,
            'biggoose': 54,
            'hagopeun': 59,
            'hydration': 60,
            'bischu': 35,
            'nus': 20,
            'fury': 24,
            'fissure': 37,
            'birdring': 48,
            'asher': 32,
            'fragi': 29,
            'shadowburn': 76,
            'carpe': 32,
            'dayfly': 37,
            'poko': 29,
            'boombox': 55,
            'profit': 33,
            'gesture': 43,
            'bdosin': 38,
            'diya': 22,
            'bani': 24,
            'coolmatt': 56,
            'undead': 43,
            'mg': 17,
            'freefeel': 43,
            'jake': 24,
            'roshan': 43,
            'altering': 48,
            'rawkus': 43,
            'agilities': 44,
            'linkzr': 36,
            'boink': 32,
            'muma': 33,
            'verbo': 34,
            'hotba': 34,
            'neptuno': 48,
            'fiveking': 47,
            'note': 26,
            'xushu': 38,
            'clockwork': 67
            }


expected_nameplates = {
    'cocco': 160,
    'cwoosh': 148,
    'mickie': 163,
    'effect': 164,
    'logix': 165,
    'custa': 165,
    'harryhook': 129,
    'tviq': 173,
    'zebbosai': 147,
    'zuppeh': 158,
    'chipshajen': 134,
    'manneten': 138,
    'mano': 162,
    'meko': 166,
    'libero': 162,
    'ark': 176,
    'saebyeolbe': 134,
    'fleta': 169,
    'pine': 174,
    'kuki': 173,
    'ryujehong': 132,
    'wekeed': 156,
    'zunba': 160,
    'tobi': 174,
    'seagull': 150,
    'xqc': 174,
    'janus': 163,
    'jjonak': 154,
    'neko': 167,
    'gamsu': 159,
    'dreamkazper': 120,
    'striker': 159,
    'kalios': 160,
    'kellex': 162,
    'taimou': 156,
    'silkthread': 137,
    'unkoe': 160,
    'soon': 165,
    'fate': 175,
    'envy': 169,
    'kariv': 165,
    'shaz': 169,
    'surefour': 144,
    'rascal': 158,
    'iremiix': 159,
    'biggoose': 142,
    'hagopeun': 137,
    'hydration': 136,
    'bischu': 161,
    'nus': 176,
    'fury': 172,
    'fissure': 158,
    'birdring': 147,
    'asher': 164,
    'fragi': 166,
    'shadowburn': 120,
    'carpe': 162,
    'dayfly': 158,
    'poko': 166,
    'boombox': 140,
    'profit': 162,
    'gesture': 153,
    'bdosin': 157,
    'diya': 174,
    'bani': 173,
    'coolmatt': 140,
    'undead': 152,
    'mg': 180,
    'freefeel': 154,
    'jake': 172,
    'roshan': 152,
    'altering': 148,
    'rawkus': 152,
    'agilities': 152,
    'linkzr': 160,
    'boink': 163,
    'muma': 163,
    'verbo': 163,
    'hotba': 162,
    'neptuno': 148,
    'fiveking': 148,
    'note': 170,
    'xushu': 161,
    'clockwork': 129,
    'munchkin': 138,
    'winz': 168,
    'nicogdh': 146,
    'akm': 172,
    'knoxxx': 150,
    'bunny': 160,
    'miro': 169
}

portrait_width = 36
plate_margin = 3
left_margin = 6
right_margin = 9

def analyze_names():
    mean_diff = 0
    mean_plate_diff = 0
    for k, v in expected.items():
        value = 0
        for c in k:
            value += CHAR_VALUES[c]
        value += len(k) - 1
        nameplate = value +portrait_width +plate_margin+ left_margin+ right_margin
        diff = v - value
        expected_nameplate = 250 - expected_nameplates[k]
        plate_diff = nameplate - expected_nameplate
        mean_diff += abs(diff) / len(expected)
        mean_plate_diff += abs(plate_diff) / len(expected)
        if diff != 0:
            print('name', k, diff)
        if plate_diff != 0:
            print('plate', k, plate_diff, nameplate, expected_nameplates[k])
    print(mean_diff)
    print(mean_plate_diff)


if __name__ == '__main__':
    main()
    #analyze_names()
