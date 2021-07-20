import pandas as pd 
from joblib import Parallel, delayed
import wget
import sox
import shutil
import os


def audio_duration(wav_path):
    return sox.file_info.duration(wav_path)


def download_data(url):
    if not os.path.isdir('data'):
        os.makedirs('data')
        wget.download(url, out='data')
        shutil.unpack_archive('data/lid_data.zip', 'data')
    else:
        print("Data already exists")


if __name__ == "__main__":
    download_data('https://storage.googleapis.com/vakyaansh-open-models/lid_data/lid_data.zip')
