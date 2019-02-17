from annotator.api_requests import get_map_modes, get_maps, get_player_list, get_color_list, get_npc_list, \
    get_hero_list, get_ability_list, get_spectator_modes, get_status_types


def get_character_set(player_set):
    chars = set()
    for p in player_set:
        chars.update(p)
    return sorted(chars)


MAP_SET = get_maps()

HERO_SET = get_hero_list() + get_npc_list()
HERO_ONLY_SET = get_hero_list()

NPC_MARKED_SET = [x + '_npc' for x in get_npc_list()]

ABILITY_SET = sorted(get_ability_list())

COLOR_SET = get_color_list()

SPECTATOR_MODES = get_spectator_modes()

PLAYER_SET = get_player_list()

STATUS_SET = get_status_types()

PLAYER_CHARACTER_SET = get_character_set(PLAYER_SET)

MAP_MODE_SET = get_map_modes()
