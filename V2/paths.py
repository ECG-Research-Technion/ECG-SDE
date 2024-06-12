from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
CONFIG_VQVAE = BASE_DIR / 'config_vqvae.json'

def load_configurations():
    with CONFIG_VQVAE.open('r') as f:
        configurations = json.load(f)
    return configurations

config = load_configurations()
predictions = config["forecast_predictions"]


# v100 is the H5 dataset, for {predictions} prediction, with 150 signals per patient, with 23040/<window size> segments per sample, with 128*5 window size
train_H5_v100 = f'train_H5_{predictions}_v100.pt'
test_H5_v100 = f'test_H5_{predictions}_v100.pt'
results_H5_dir_v100 = f"results_H5_{predictions}_v100"
checkpoints_H5_dir_v100 = f"checkpoints_H5_{predictions}_v100"

datasets_dir_H5 = 'H5'

# v110 is the MITBIH dataset, with transformer, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v110 = f'train_MITBIH_{predictions}_v110.pt'
test_MITBIH_v110 = f'test_MITBIH_{predictions}_v110.pt'
results_dir_MITBIH_v110 = f"results_MITBIH_{predictions}_v110"
checkpoints_dir_MITBIH_v110 = f"checkpoints_MITBIH_{predictions}_v110"

# v111 is the MITBIH dataset, no transformer, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v111 = f'train_MITBIH_{predictions}_v111.pt'
test_MITBIH_v111 = f'test_MITBIH_{predictions}_v111.pt'
results_dir_MITBIH_v111 = f"results_MITBIH_{predictions}_v111"
checkpoints_dir_MITBIH_v111 = f"checkpoints_MITBIH_{predictions}_v111"

# v112 is the MITBIH dataset, with transformer, for {predictions} prediction, with 1 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v112 = f'train_MITBIH_{predictions}_v112.pt'
test_MITBIH_v112 = f'test_MITBIH_{predictions}_v112.pt'
results_dir_MITBIH_v112 = f"results_MITBIH_{predictions}_v112"
checkpoints_dir_MITBIH_v112 = f"checkpoints_MITBIH_{predictions}_v112"

# v113 is the MITBIH dataset, with transformer, LeakyReLU AL, for {predictions} prediction, with 1 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v113 = f'train_MITBIH_{predictions}_v112.pt'
test_MITBIH_v113 = f'test_MITBIH_{predictions}_v112.pt'
results_dir_MITBIH_v113 = f"results_MITBIH_{predictions}_v113"
checkpoints_dir_MITBIH_v113 = f"checkpoints_MITBIH_{predictions}_v113"

# v114 is the MITBIH dataset, w/o transformer, LeakyReLU AL, 2048 codebock and vector size in quantization, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v114 = f'train_MITBIH_{predictions}_v111.pt'
test_MITBIH_v114 = f'test_MITBIH_{predictions}_v111.pt'
results_dir_MITBIH_v114 = f"results_MITBIH_{predictions}_v114"
checkpoints_dir_MITBIH_v114 = f"checkpoints_MITBIH_{predictions}_v114"

# v115 is the MITBIH dataset, w/o transformer, Norm batch -> SiLU AL, 1024 codebock and vector size in quantization, 0.75 MSE loss and 0.25 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v115 = f'train_MITBIH_{predictions}_v111.pt'
test_MITBIH_v115 = f'test_MITBIH_{predictions}_v111.pt'
results_dir_MITBIH_v115 = f"results_MITBIH_{predictions}_v115"
checkpoints_dir_MITBIH_v115 = f"checkpoints_MITBIH_{predictions}_v115"

# v001 is the MITBIH dataset, w/o transformer, Norm batch -> SiLU AL, 1024 codebock and vector size in quantization, 0.75 MSE loss and 0.25 latent loss, for {predictions} prediction, 1 signal, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v001 = f'train_MITBIH_{predictions}_v001.pt'
test_MITBIH_v001 = f'test_MITBIH_{predictions}_v001.pt'
results_dir_MITBIH_v001 = f"results_MITBIH_{predictions}_v001"
checkpoints_dir_MITBIH_v001 = f"checkpoints_MITBIH_{predictions}_v001"

# v116 is the MITBIH dataset, w/o transformer, Norm batch -> SiLU AL, 512 vector size in quantization, 0.8 MSE loss and 0.2 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v116 = f'train_MITBIH_{predictions}_v111.pt'
test_MITBIH_v116 = f'test_MITBIH_{predictions}_v111.pt'
results_dir_MITBIH_v116 = f"results_MITBIH_{predictions}_v116"
checkpoints_dir_MITBIH_v116 = f"checkpoints_MITBIH_{predictions}_v116"

datasets_dir_MITBIH = 'MITBIH'

# current settings
train_data = train_MITBIH_v116
test_data = test_MITBIH_v116
results_dir = results_dir_MITBIH_v116
checkpoints_dir = checkpoints_dir_MITBIH_v116
datasets_dir = datasets_dir_MITBIH


RESULTS_DIR = BASE_DIR / 'results'
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
DATASETS_DIR = BASE_DIR.parent / 'datasets' / datasets_dir_MITBIH
TRAIN_DATA = DATASETS_DIR / train_data
TEST_DATA = DATASETS_DIR / test_data
RESULTS_DIR = RESULTS_DIR / results_dir
CHECKPOINTS_DIR = CHECKPOINTS_DIR / checkpoints_dir