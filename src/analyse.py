from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import mne
import warnings

warnings.filterwarnings('ignore')

processes = 10


def _get_channels(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    return raw.ch_names


def _compute_channel_distribution_for_dir(data_path):
    with Pool(processes=processes) as p:
        res = p.map(_get_channels, data_path.resolve().iterdir())
    all_channels = []
    for r in res:
        all_channels.extend(r)
    return Counter(all_channels)


def _update_channel_distribution(channel_distribution, other_distribution):
    for ch, count in other_distribution.items():
        if ch in channel_distribution:
            channel_distribution[ch] += count
        else:
            channel_distribution[ch] = count


def compute_channel_distribution(data_path):
    train_abnormal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/train/abnormal/01_tcp_ar")
    ta_res = _compute_channel_distribution_for_dir(train_abnormal)
    train_normal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/train/normal/01_tcp_ar")
    tn_res = _compute_channel_distribution_for_dir(train_normal)
    eval_abnormal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/eval/abnormal/01_tcp_ar")
    ea_res = _compute_channel_distribution_for_dir(eval_abnormal)
    eval_normal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/eval/normal/01_tcp_ar")
    en_res = _compute_channel_distribution_for_dir(eval_normal)
    channel_distribution = ta_res.copy()
    _update_channel_distribution(channel_distribution, tn_res)
    _update_channel_distribution(channel_distribution, ea_res)
    _update_channel_distribution(channel_distribution, en_res)
    return channel_distribution


if __name__ == '__main__':
    channel_distribution = compute_channel_distribution('/home/denis/Загрузки/TUAB')
    channel_df = pd.DataFrame(channel_distribution.items(), columns=["Channel", "Count"])
    channel_df = channel_df.sort_values(by="Count", ascending=False)
    plt.figure(figsize=(20, 5))
    plt.bar(channel_df["Channel"], channel_df["Count"], color="skyblue")
    plt.xticks(rotation=90)
    plt.xlabel("Channel")
    plt.ylabel("Freq")
    plt.title("Channel freq distribution")
    plt.tight_layout()
    plt.savefig("results/tuab_data_channel_freq.png")
    plt.show()
