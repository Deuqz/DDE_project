import time
from prefect import flow, task

from snr import calculate_snr, aggregate_snr
from database import init_database
import subprocess


@task
def download_tuab(time=None):
    try:
        print("Start download TUAB")
        start_time = time.time()
        rc = subprocess.call("src/download.sh", shell=True)
        elapsed_time = time.time() - start_time
        print(f"TUAB was downloaded. Elapsed time: {elapsed_time}")
    except Exception as e:
        print("Error while downloading")
        raise e


@task
def check_quality():
    try:
        print("Start calculate snr")
        start_time = time.time()
        calculate_snr('/home/denis/Загрузки/TUAB')
        print("Got snr:", aggregate_snr())
        elapsed_time = time.time() - start_time
        print(f"End calculate snr. Elapsed time: {elapsed_time}")
    except Exception as e:
        print("Error while check quality")
        raise e


@task
def create_database():
    try:
        print("Starting creating database")
        start_time = time.time()
        init_database('/home/denis/Загрузки/TUAB')
        elapsed_time = time.time() - start_time
        print(f"Database created. Elapsed time: {elapsed_time}")
    except Exception as e:
        print("Error while creating database")
        raise e


@flow
def data_processing_workflow():
    workflow_start_time = time.time()
    print("Workflow started")

    download_tuab()

    check_quality()

    create_database()

    workflow_elapsed_time = time.time() - workflow_start_time
    print(f"Workflow completed successfully. Elapsed time: {workflow_elapsed_time}")


if __name__ == "__main__":
    data_processing_workflow()
