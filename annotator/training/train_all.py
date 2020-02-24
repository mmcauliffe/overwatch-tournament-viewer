import os
import subprocess


script_dir = os.path.dirname(os.path.abspath(__file__))
data_gen_dir = os.path.join(os.path.dirname(script_dir), 'data_generation')
annotate_dir = os.path.join(os.path.dirname(script_dir), 'annotator')
validation_dir = os.path.join(script_dir, 'validation_errors')
fine_tune_dir = os.path.join(script_dir, 'fine_tune')
python_exe =r'C:\Users\micha\Miniconda3\envs\torch\python.exe'

annotate_scripts = [
    #'inout_game.py',
    #'annotate_round_events.py',
]

base_train_scripts = [
    #'train_kf_exist_torch_cnn.py',
    #'train_base_kf_slot_ctc_torch.py',
    #'train_game_torch_cnn.py',
]

data_generation_scripts = [
    'generate_data.py',
    'generate_game_data.py',
    #'generate_ocr_data.py',
]

train_scripts = [
    #'train_player_name_ocr_torch.py',
    #'train_player_status_torch_cnn.py',
    #'train_kf_slot_ctc_torch.py',
    #'train_game_torch_cnn.py',
    #'train_game_detail_torch_cnn.py',
    #'train_kf_exist_torch_cnn.py',
    #'train_base_kf_slot_ctc_torch.py',
    #'train_mid_torch_cnn.py',
]

fine_tune_scripts = [
    #'fine_tune_kf_exist.py',
    #'fine_tune_player_status.py',
    #'fine_tune_base_kf_slot.py',
    'fine_tune_game.py',
    #'fine_tune_mid.py',
    'fine_tune_kf_slot.py',
]

test_scripts = [
    #'test_kf_exists.py',
    #'test_kf_base.py',
    #'test_game_detail.py',
    #'test_ocr.py',
    #'test_player_status.py',
    #'test_kf.py',
    #'test_game.py',
    #'test_mid.py',
]

## BEGIN

for s in annotate_scripts:
    path = os.path.join(annotate_dir, s)

    subprocess.call([python_exe, path])

for s in base_train_scripts:
    path = os.path.join(script_dir, s)

    subprocess.call([python_exe, path])


for s in data_generation_scripts:
    path = os.path.join(data_gen_dir, s)

    subprocess.call([python_exe, path])

for s in train_scripts:
    path = os.path.join(script_dir, s)

    subprocess.call([python_exe, path])

for s in fine_tune_scripts:
    path = os.path.join(fine_tune_dir, s)
    subprocess.call([python_exe, path])


for s in test_scripts:
    path = os.path.join(validation_dir, s)

    subprocess.call([python_exe, path])