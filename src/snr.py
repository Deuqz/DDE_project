import numpy as np
import mne
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import json
import warnings

warnings.filterwarnings('ignore')

path_to_save = Path("results/snr")
processes = 10

def compute_snr_realization(raw):
    spectrum = raw.compute_psd(method="welch", verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)
    noise_n_neighbor_freqs = 3
    noise_skip_neighbor_freqs = 1

    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs)))
    averaging_kernel /= averaging_kernel.sum()

    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
        axis=-1, arr=psd
    )
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    psd = psd[:, edge_width:-edge_width]

    snr = psd / mean_noise
    snr[np.logical_not(np.isfinite(snr))] = 0
    return snr


def compute_snr(data_path, save_dir):
    i, data_path = data_path
    if i % 10 == 0:
        print("\033[A                             \033[A")
        print(i)
    raw = mne.io.read_raw_edf(data_path, preload=True, verbose=False)
    snr = compute_snr_realization(raw)
    snr_max = snr.max(axis=1)
    if snr_max is not None:
        if snr_max.sort() is None:
            snr_max_top5 = None
        else:
            snr_max_top5 = snr_max.sort()[-5:]
        snr_max_mean = snr_max.mean()
        save_file = save_dir / Path(data_path.name[:-4] + '.json')
        with save_file.open('w') as f:
            json.dump({'base_path': str(data_path), 'snr_max_top5': snr_max_top5, 'snr_max_mean': snr_max_mean}, f)


def _process_part(from_path, to_path):
    to_path.mkdir(parents=True, exist_ok=True)
    with Pool(processes=processes) as p:
        p.map(partial(compute_snr, save_dir=to_path), enumerate(from_path.resolve().iterdir()))


def calculate_snr(data_path):
    print()
    train_abnormal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/train/abnormal/01_tcp_ar")
    train_abnormal_save_dir = path_to_save / Path("train_abnormal")
    _process_part(train_abnormal, train_abnormal_save_dir)

    train_normal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/train/normal/01_tcp_ar")
    train_normal_save_dir = path_to_save / Path("train_normal")
    _process_part(train_normal, train_normal_save_dir)

    eval_abnormal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/eval/abnormal/01_tcp_ar")
    eval_abnormal_save_dir = path_to_save / Path("eval_abnormal")
    _process_part(eval_abnormal, eval_abnormal_save_dir)

    eval_normal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/eval/normal/01_tcp_ar")
    eval_normal_save_dir = path_to_save / Path("eval_normal")
    _process_part(eval_normal, eval_normal_save_dir)


def aggregate_snr():
    result = {}
    for dir_path in path_to_save.resolve().iterdir():
        snr_max_mean = []
        for file_path in dir_path.iterdir():
            data = json.load(file_path.open('r'))
            if np.isfinite(data['snr_max_mean']):
                snr_max_mean.append(data['snr_max_mean'])
        result[dir_path.name] = {'snr_max_mean': np.array(snr_max_mean).mean()}
    return result


if __name__ == '__main__':
    calculate_snr('/home/denis/Загрузки/TUAB')
    print(aggregate_snr())
