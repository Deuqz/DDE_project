import pickle
import time
import uuid
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import random
from typing import List

import mne.io
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import src.preprocessing as preprocessing
import src.snr as snr

engine = create_engine('sqlite:///eeg.db', echo=True)
Base = declarative_base()


class TrainTuab(Base):
    __tablename__ = 'train_tuab'

    id = Column(String, primary_key=True)
    file_name = Column(String, nullable=False)
    window_id = Column(Integer, nullable=False)
    ch_names = Column(LargeBinary, nullable=False)
    label = Column(Integer, nullable=False)
    eeg = Column(LargeBinary, nullable=False)


class EvalTuab(Base):
    __tablename__ = 'eval_tuab'

    id = Column(String, primary_key=True)
    file_name = Column(String, nullable=False)
    window_id = Column(Integer, nullable=False)
    ch_names = Column(LargeBinary, nullable=False)
    label = Column(Integer, nullable=False)
    eeg = Column(LargeBinary, nullable=False)


class Raw(Base):
    __tablename__ = 'raw'

    id = Column(String, primary_key=True)
    file_name = Column(String, nullable=False)
    eeg = Column(LargeBinary, nullable=False)


Base.metadata.create_all(engine)


def _update_table(file_paths, session, constructor, procs=5):
    with Pool(processes=procs) as p:
        for info in p.map(preprocessing.get_data, file_paths):
            for i, window in enumerate(info['windows']):
                session.add(constructor(
                    id=str(uuid.uuid4()),
                    file_name=info['file_name'],
                    window_id=i,
                    ch_names=pickle.dumps(info['ch_names']),
                    label=info['label'],
                    eeg=pickle.dumps(window)
                ))


def _update_raw_table(file_paths, session, procs=5):
    with Pool(processes=procs) as p:
        for raw in p.map(partial(mne.io.read_raw_edf, preload=True, verbose=False), file_paths):
            session.add(Raw(
                id=str(uuid.uuid4()),
                file_name=raw.filenames[0].name,
                eeg=pickle.dumps(raw)
            ))


def init_database(data_path):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    train_abnormal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/train/abnormal/01_tcp_ar")
    train_normal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/train/normal/01_tcp_ar")
    train_abnormal_paths = [(0, pth) for pth in train_abnormal.iterdir()]
    train_normal_paths = [(1, pth) for pth in train_normal.iterdir()]
    train_paths = train_abnormal_paths + train_normal_paths
    random.shuffle(train_paths)
    _update_table(train_paths[:5], session, TrainTuab)

    session.commit()

    eval_abnormal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/eval/abnormal/01_tcp_ar")
    eval_normal = Path(data_path) / Path("tuh_eeg_abnormal/v3.0.1/edf/eval/normal/01_tcp_ar")
    eval_abnormal_paths = [(0, pth) for pth in eval_abnormal.iterdir()]
    eval_normal_paths = [(1, pth) for pth in eval_normal.iterdir()]
    eval_paths = eval_abnormal_paths + eval_normal_paths
    random.shuffle(eval_paths)
    _update_table(eval_paths[:5], session, EvalTuab)

    session.commit()

    paths = [p for _, p in train_paths] + [p for _, p in eval_paths]
    _update_raw_table(paths[:5], session)

    session.commit()


def update_database(dir_paths: List[str], labels: List[int], type: str):
    Session = sessionmaker(bind=engine)
    session = Session()

    all_paths = []
    for path, label in zip(dir_paths, labels):
        all_paths.extend([(label, p) for p in Path(path).iterdir()])
    random.shuffle(all_paths)
    constructor = TrainTuab if type == 'train' else EvalTuab
    _update_table(all_paths, session, constructor)

    session.commit()

    _update_raw_table(all_paths, session)

    session.commit()


def recalculate_metrics():
    Session = sessionmaker(bind=engine)
    session = Session()

    snrs_max = []
    for sample in session.execute(select(Raw)):
        snr_res = snr.compute_snr_realization(pickle.loads(sample[0].eeg))
        snr_max = snr_res.max(axis=1).mean()
        snrs_max.append(snr_max)

    return np.mean(snrs_max)


if __name__ == '__main__':
    init_database('/home/denis/Загрузки/TUAB')
    time.sleep(1000000000)
    recalculate_metrics()
