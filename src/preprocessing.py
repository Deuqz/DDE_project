from pathlib import Path

import mne


channel_order = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
                'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF',
                'EEG T1-REF', 'EEG T2-REF']

channel_set = set(channel_order)

rename_map = {'EEG FP1-REF': 'FP1', 'EEG FP2-REF': 'FP2', 'EEG F3-REF': 'F3', 'EEG F4-REF': 'F4', 'EEG C3-REF': 'C3',
              'EEG C4-REF': 'C4', 'EEG P3-REF': 'P3', 'EEG P4-REF': 'P4', 'EEG O1-REF': 'O1', 'EEG O2-REF': 'O2',
              'EEG F7-REF': 'F7', 'EEG F8-REF': 'F8', 'EEG T3-REF': 'T3', 'EEG T4-REF': 'T4', 'EEG T5-REF': 'T5',
              'EEG T6-REF': 'T6', 'EEG A1-REF': 'A1', 'EEG A2-REF': 'A2', 'EEG FZ-REF': 'FZ', 'EEG CZ-REF': 'CZ',
              'EEG PZ-REF': 'PZ', 'EEG T1-REF': 'T1', 'EEG T2-REF': 'T2'}


def preprocess_file(raw):
    drop_ch = []
    for ch in raw.ch_names:
        if ch not in channel_set:
            drop_ch.append(ch)
    raw.drop_channels(drop_ch)
    if len(raw.ch_names) != len(channel_order):
        return None
    raw.reorder_channels(channel_order)
    raw.rename_channels(rename_map)
    raw.filter(l_freq=0.1, h_freq=75.0, verbose=False)
    raw.notch_filter(50.0, verbose=False)
    raw.resample(200)
    windows = []
    raw_data = raw.get_data(units='uV')
    window_size = 2000
    for i in range(raw_data.shape[1] // window_size):
        windows.append(raw_data[:, i * window_size : (i + 1) * window_size])
    return {
        'windows': windows,
        'ch_names': raw.ch_names
    }


def get_data(file_path):
    if isinstance(file_path, tuple):
        label, file_path = file_path
    else:
        label = None
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    preprocessed_edf = preprocess_file(raw)
    preprocessed_edf['file_name'] = Path(file_path).resolve().name
    preprocessed_edf['label'] = label
    return preprocessed_edf
