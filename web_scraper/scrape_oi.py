import os
import annotator
import json
from annotator.api_requests import get_matches, get_match_stats

output_directory = r'N:\Data\Overwatch\oi'

os.makedirs(output_directory, exist_ok=True)

event = 122


if __name__ == '__main__':
    matches = get_matches(event)
    for m in matches:
        print(m)
        event_name = m['event']
        event_directory = os.path.join(output_directory, event_name)
        os.makedirs(event_directory, exist_ok=True)
        stats = get_match_stats(m['id'])
        m_path = os.path.join(event_directory, '{} vs {} - {} - {}.json'.format(stats['team_one'].replace('?', ''),
                                                                                stats['team_two'].replace('?', ''), stats['date'], m['id']))
        if os.path.exists(m_path):
            continue
        with open(m_path, 'w', encoding='utf8') as f:
            json.dump(stats, f, indent=4)