import pickle
from collections import Counter

import pandas as pd
import streamlit as st
import mne
import matplotlib.pyplot as plt
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from database import engine, Raw, recalculate_metrics, TrainTuab, EvalTuab
from src.preprocessing import channel_order

st.set_page_config(page_title="EEG Dashboard", layout="wide")


constructor_map = {
    'train_tuab': TrainTuab,
    'eval_tuab': EvalTuab,
    'raw': Raw
}

def load_edf(file):
    raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
    return raw


def analyse_eeg(file_id: str):
    Session = sessionmaker(bind=engine)
    session = Session()
    raw = None
    for x in session.execute(select(Raw).where(Raw.id == file_id)):
        raw = pickle.loads(x[0].eeg)
    if raw is None:
        return

    st.sidebar.subheader("Параметры визуализации")
    selected_channel = st.sidebar.selectbox(
        "Выберите канал для анализа", channel_order
    )

    st.subheader("Информация о файле")
    st.write(f"Частота дискретизации: {raw.info['sfreq']} Гц")
    st.write(f"Количество каналов: {len(raw.info['ch_names'])}")
    st.write(f"Длительность записи: {raw.n_times / raw.info['sfreq']:.2f} секунд")
    st.write("Каналы:")
    st.write(", ".join(raw.info['ch_names']))

    st.subheader(f"Временной сигнал - {selected_channel}")
    channel_data, times = raw[selected_channel, :]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, channel_data.T, lw=0.5)
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Амплитуда (В)")
    ax.set_title(f"Временной сигнал для канала {selected_channel}")
    st.pyplot(fig)

    st.subheader(f"Спектральная плотность мощности (PSD) - {selected_channel}")
    fig, ax = plt.subplots(figsize=(10, 4))
    spectrum = raw.compute_psd(method='welch', picks=[selected_channel], fmin=0.5, fmax=50, verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)
    ax.plot(freqs, psd[0], lw=0.8)
    ax.set_xlabel("Частота (Гц)")
    ax.set_ylabel("Мощность (dB)")
    ax.set_title(f"PSD для канала {selected_channel}")
    st.pyplot(fig)


def main():
    st.title("Temple University Abnormal Corpus (TUAB)")

    st.sidebar.header("Введите id eeg")
    file_id = st.sidebar.chat_input()

    table_name = st.selectbox(
        "Select a table",
        ["train_tuab", "eval_tuab", "raw"]
    )

    if file_id:
        analyse_eeg(file_id)
    elif table_name:
        query = f"SELECT * FROM {table_name} LIMIT 5"
        with engine.connect() as connection:
            data = pd.read_sql_query(query, connection)

        data['eeg'] = data['eeg'].apply(lambda x: x[:100])
        if 'ch_names' in data:
            data['ch_names'] = data['ch_names'].apply(lambda x: pickle.loads(x))

        st.write(f"Первые 5 строк `{table_name}`:")
        st.dataframe(data.head(5), use_container_width=True)

        if table_name == 'raw':
            m = recalculate_metrics()
            st.metric('Mean of maximum snr by channels', m)

        st.subheader(f"Распределение каналов")

        Session = sessionmaker(bind=engine)
        session = Session()

        constructor = constructor_map[table_name]
        all_ch_names = []
        for sample in session.execute(select(constructor)):
            sample = sample[0]
            if table_name == 'raw':
                raw = pickle.loads(sample.eeg)
                all_ch_names.extend(raw.ch_names)
            else:
                all_ch_names.extend(pickle.loads(sample.ch_names))
        distribution = Counter(all_ch_names)
        channel_df = pd.DataFrame(distribution.items(), columns=["Channel", "Count"])
        channel_df = channel_df.sort_values(by="Count", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(channel_df["Channel"], channel_df["Count"], color="skyblue")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Freq")
        ax.set_title("Channel freq distribution")
        st.pyplot(fig)
    else:
        pass


if __name__ == "__main__":
    main()
