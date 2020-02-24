import os
import subprocess


script_dir = os.path.dirname(os.path.abspath(__file__))
annotate_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'annotator')
data_gen_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'data_generation')
python_exe =r'C:\Users\micha\Miniconda3\envs\torch\python.exe'

data_generation_scripts = [
    #'generate_data.py'
]

scripts = [
    #'test_game.py',
    'test_player_status.py',
    'test_kf.py',
    #'test_ocr.py',
]

for s in data_generation_scripts:
    path = os.path.join(data_gen_dir, s)

    subprocess.call([python_exe, path])


for s in scripts:
    path = os.path.join(script_dir, s)

    subprocess.call([python_exe, path])

annotate_scripts = [
    #'annotate_round_events.py',
    #'inout_game.py'
]

for s in annotate_scripts:
    path = os.path.join(annotate_dir, s)

    subprocess.call([python_exe, path])