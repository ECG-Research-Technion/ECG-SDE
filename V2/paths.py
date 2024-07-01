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

# v200 on the new arch, is the MITBIH dataset, w/o transformer, Norm batch -> LeakyReLU AL, 64 embed size in quantization, MSE loss and 0.25 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v200 = f'train_MITBIH_{predictions}_v111.pt'
test_MITBIH_v200 = f'test_MITBIH_{predictions}_v111.pt'
results_dir_MITBIH_v200 = f"results_MITBIH_{predictions}_v200"
checkpoints_dir_MITBIH_v200 = f"checkpoints_MITBIH_{predictions}_v200"

# v201 on the new arch, is the MITBIH dataset, w/o transformer, bigger model, Norm batch -> LeakyReLU AL, 100 embed size in quantization, MSE loss and 0.25 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v201 = f'train_MITBIH_{predictions}_v111.pt'
test_MITBIH_v201 = f'test_MITBIH_{predictions}_v111.pt'
results_dir_MITBIH_v201 = f"results_MITBIH_{predictions}_v201"
checkpoints_dir_MITBIH_v201 = f"checkpoints_MITBIH_{predictions}_v201"

# v202 on the new arch, is the MITBIH dataset, w/o transformer, bigger model, bigger window-size, LeakyReLU AL, 64 embed size in quantization, MSE loss and 0.25 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v202 = f'train_MITBIH_{predictions}_v111.pt'
test_MITBIH_v202 = f'test_MITBIH_{predictions}_v111.pt'
results_dir_MITBIH_v202 = f"results_MITBIH_{predictions}_v202"
checkpoints_dir_MITBIH_v202 = f"checkpoints_MITBIH_{predictions}_v202"

# v203 on the new arch, is the MITBIH dataset, w/o transformer, bigger+ model, bigger window-size, LeakyReLU AL, 64 embed size in quantization, MSE loss and 0.25 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v203 = f'train_MITBIH_{predictions}_v111.pt'
test_MITBIH_v203 = f'test_MITBIH_{predictions}_v111.pt'
results_dir_MITBIH_v203 = f"results_MITBIH_{predictions}_v203"
checkpoints_dir_MITBIH_v203 = f"checkpoints_MITBIH_{predictions}_v203"

# v204 on the new arch, is the MITBIH dataset, w/o transformer, bigger+ model + batchnorm, bigger window-size, LeakyReLU AL, 64 embed size in quantization, MSE loss and 0.25 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v204 = f'train_MITBIH_{predictions}_v111.pt'
test_MITBIH_v204 = f'test_MITBIH_{predictions}_v111.pt'
results_dir_MITBIH_v204 = f"results_MITBIH_{predictions}_v204"
checkpoints_dir_MITBIH_v204 = f"checkpoints_MITBIH_{predictions}_v204"

# v205 on the new arch, is the MITBIH dataset, model cnn is focused on sequence length LeakyReLU AL, 32 embed size in quantization, MSE loss and 0.25 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v205 = f'train_MITBIH_{predictions}_5sec_50sig.pt'
test_MITBIH_v205 = f'test_MITBIH_{predictions}_5sec_50sig.pt'
results_dir_MITBIH_v205 = f"results_MITBIH_{predictions}_v205"
checkpoints_dir_MITBIH_v205 = f"checkpoints_MITBIH_{predictions}_v205"

# v210 on the new arch based on cnn+tanh, is the MITBIH dataset, 64 embed size in quantization, MSE loss and 0.25 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v210 = f'train_MITBIH_{predictions}_5sec_50sig.pt'
test_MITBIH_v210 = f'test_MITBIH_{predictions}_5sec_50sig.pt'
results_dir_MITBIH_v210 = f"results_MITBIH_{predictions}_v210"
checkpoints_dir_MITBIH_v210 = f"checkpoints_MITBIH_{predictions}_v210"

# v211 on the new arch based on cnn+tanh, is the MITBIH dataset, bigger+ model + 1 LayerNorm, 64 embed size in quantization, MSE loss and 0.25 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v211 = f'train_MITBIH_{predictions}_5sec_50sig.pt'
test_MITBIH_v211 = f'test_MITBIH_{predictions}_5sec_50sig.pt'
results_dir_MITBIH_v211 = f"results_MITBIH_{predictions}_v211"
checkpoints_dir_MITBIH_v211 = f"checkpoints_MITBIH_{predictions}_v211"

# v212 on the new arch based on cnn+tanh, is the MITBIH dataset, bigger+ model, 64 embed size in quantization, MSE loss and 0.25 latent loss, for {predictions} prediction, with 100 signals per patient, with 3 input signals duration seconds, with 360 sample per second
train_MITBIH_v212 = f'train_MITBIH_{predictions}_5sec_50sig.pt'
test_MITBIH_v212 = f'test_MITBIH_{predictions}_5sec_50sig.pt'
results_dir_MITBIH_v212 = f"results_MITBIH_{predictions}_v212"
checkpoints_dir_MITBIH_v212 = f"checkpoints_MITBIH_{predictions}_v212"

datasets_dir_MITBIH = 'MITBIH'

# current settings
train_data = train_MITBIH_v212
test_data = test_MITBIH_v212
results_dir = results_dir_MITBIH_v212
checkpoints_dir = checkpoints_dir_MITBIH_v212
datasets_dir = datasets_dir_MITBIH


RESULTS_DIR = BASE_DIR / 'results'
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
DATASETS_DIR = BASE_DIR.parent / 'datasets' / datasets_dir_MITBIH
TRAIN_DATA = DATASETS_DIR / train_data
TEST_DATA = DATASETS_DIR / test_data
RESULTS_DIR = RESULTS_DIR / results_dir
CHECKPOINTS_DIR = CHECKPOINTS_DIR / checkpoints_dir