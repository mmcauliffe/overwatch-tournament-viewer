

txt = ''''''

status_rounds = [int(x) for x in txt.split('\n') if x]

txt = '''15461
15455
15444
15442
15440
15424
15431
15468
15448
15447
15444
15440
15431
15429
15424'''

kf_rounds = [int(x) for x in txt.split('\n') if x]

spec_modes = [
    'original',
    'overwatch league',
    'overwatch league season 3',
    'world cup',
    'contenders']

status_train_dir = r'N:\Data\Overwatch\training_data\player_status'
kf_train_dirs = [r'N:\Data\Overwatch\training_data\kill_feed_ctc',
                 r'N:\Data\Overwatch\training_data\kill_feed_ctc_base']
import os

for m in spec_modes:
    m_dir = os.path.join(status_train_dir, m)
    if os.path.exists(m_dir):
        for f in os.listdir(m_dir):
            round_id = int(os.path.splitext(f)[0])
            if round_id in status_rounds:
                print('deleting {}'.format(round_id))
                os.remove(os.path.join(m_dir, f))
    for kf_train_dir in kf_train_dirs:
        m_dir = os.path.join(kf_train_dir, m)
        if os.path.exists(m_dir):
            for f in os.listdir(m_dir):
                round_id = int(os.path.splitext(f)[0].replace('_exists', ''))
                if round_id in kf_rounds:
                    print('deleting {}'.format(os.path.join(m_dir, f)))
                    os.remove(os.path.join(m_dir, f))