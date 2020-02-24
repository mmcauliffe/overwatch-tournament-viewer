import os

txt = '''11819
11647
10324
10299'''

rounds = [int(x) for x in txt.split('\n') if x]

train_dir = r'N:\Data\Overwatch\training_data\player_ocr'

for f in os.listdir(train_dir):
    if not f.endswith('.hdf5'):
        continue
    round_id = int(os.path.splitext(f)[0])
    if round_id in rounds:
        print('deleting {}'.format(round_id))
        os.remove(os.path.join(train_dir, f))