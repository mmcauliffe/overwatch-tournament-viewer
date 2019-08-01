import os
import subprocess

script_dir = os.path.dirname(os.path.basename(__file__))
python_exe =r'C:\Users\micha\Miniconda3\envs\torch\python.exe'

scripts = [
    #'train_game_torch_cnn.py',
    'train_kf_slot_ctc_torch.py',
    'train_mid_torch_cnn.py',
    #'train_pause_torch_cnn.py',
    'train_player_name_ocr_torch.py',
    'train_player_status_torch_cnn.py',
    #'train_replay_torch_cnn.py',
    'train_kf_exists_torch_cnn.py',
]

for s in scripts:
    path = os.path.join(script_dir, s)

    subprocess.call([python_exe, path])