import pandas as pd 
from joblib import Parallel, delayed
import wget
import sox
import shutil
import os
from glob import glob
from tqdm import tqdm


def audio_duration(wav_path):
    return sox.file_info.duration(wav_path)


def download_data(url):
    if not os.path.isdir('data'):
        os.makedirs('data')
        wget.download(url, out='data')
        shutil.unpack_archive('data/lid_data.zip', 'data')
    else:
        print("Data already exists")

def explore_audio_data():
    audio_files = glob('data/**/*.wav', recursive=True)

    audio_durations = Parallel(n_jobs=os.cpu_count())(delayed(audio_duration)(f) for f in tqdm(audio_files,
                                                        leave=False, total=len(audio_files)))

    print(f'Total duration: {sum(audio_durations)/3600 : .2f} hours')

    audio_data = pd.DataFrame({'path': audio_files, 'duration': audio_durations})

    audio_data['language'] = audio_data['path'].apply(lambda x: x.split('/')[2])

    print(audio_data.groupby(by=['language'])['duration'].sum() / 3600)

    audio_data.to_csv('data/audio_data.csv', index=False)

if __name__ == "__main__":
    download_data('https://storage.googleapis.com/vakyaansh-open-models/lid_data/lid_data.zip')
    explore_audio_data()