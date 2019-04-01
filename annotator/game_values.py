from annotator.api_requests import get_map_modes, get_maps, get_player_list, get_color_list, get_npc_list, \
    get_hero_list, get_ability_list, get_spectator_modes, get_status_types, get_train_info


def get_character_set(player_set):
    chars = set()
    for p in player_set:
        chars.update(p)
    return sorted(chars)

TRAIN_INFO = get_train_info()

MAP_SET = TRAIN_INFO['maps']

HERO_SET = TRAIN_INFO['heroes']

LABEL_SET = TRAIN_INFO['kill_feed_labels']

ABILITY_SET = sorted(get_ability_list())

COLOR_SET = TRAIN_INFO['colors']

SPECTATOR_MODES = TRAIN_INFO['spectator_modes']

#PLAYER_SET = get_player_list()

STATUS_SET = TRAIN_INFO['statuses']

#PLAYER_CHARACTER_SET = get_character_set(PLAYER_SET)

MAP_MODE_SET = TRAIN_INFO['map_modes']
