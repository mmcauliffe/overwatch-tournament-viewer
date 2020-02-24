from annotator.api_requests import get_map_modes, get_maps, get_player_list, get_color_list, get_npc_list, \
    get_hero_list, get_ability_list, get_spectator_modes, get_status_types, get_train_info, get_kill_feed_info


def get_character_set(player_set):
    chars = set()
    for p in player_set:
        chars.update(p)
    return sorted(chars)


TRAIN_INFO = get_train_info()

MAP_SET = TRAIN_INFO['maps']

SUBMAP_SET = TRAIN_INFO['submaps']

GAME_SET = TRAIN_INFO['game']

HERO_SET = TRAIN_INFO['heroes']

LABEL_SET = TRAIN_INFO['kill_feed_labels']

ABILITY_SET = sorted(get_ability_list())

KILL_FEED_INFO = get_kill_feed_info()

COLOR_SET = TRAIN_INFO['colors']

SPECTATOR_MODES = TRAIN_INFO['spectator_modes']

FILM_FORMATS = TRAIN_INFO['film_formats']

STATUS_SET = ['status'] + TRAIN_INFO['extra_statuses']

PLAYER_CHARACTER_SET = TRAIN_INFO['player_name_characters']

MAP_MODE_SET = TRAIN_INFO['map_modes']
