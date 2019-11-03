import os
import annotator
import json
print(dir(annotator))
from annotator.api_requests import get_matches, get_match_stats

output_directory = r'E:\Data\Overwatch\owl_stats\oi_api'

os.makedirs(output_directory, exist_ok=True)

event = 100

if __name__ == '__main__':
    matches = get_matches(event)
    for m in matches:
        print(m)
        stats = get_match_stats(m['id'])
        m_path = os.path.join(output_directory, '{} - {}.json'.format(stats['date'], m['name']))
        if os.path.exists(m_path):
            continue
        if not stats['maps']:
            continue
        with open(m_path, 'w', encoding='utf8') as f:
            json.dump(stats, f, indent=4)