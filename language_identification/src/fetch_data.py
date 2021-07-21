import pandas as pd 
from joblib import Parallel, delayed
import wget
import sox
import shutil
import os
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


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

    label_encoder = LabelEncoder()
    

    audio_data['language'] = audio_data['path'].apply(lambda x: x.split('/')[2])
    audio_data['label'] = label_encoder.fit_transform(audio_data['language'])
    
    print(audio_data.groupby(by=['language'])['duration'].sum() / 3600)

    audio_data.to_csv('data/audio_data.csv', index=False)

    X_train, X_test, y_train, y_test = train_test_split(audio_data['path'], audio_data['label'],
                                                        test_size=0.20, random_state=42, stratify=audio_data['label'])
    
    train_data = pd.DataFrame({'path': X_train, 'label': y_train})
    valid_data = pd.DataFrame({'path': X_test, 'label': y_test})

    train_data.to_csv('data/train.csv', index=False)
    valid_data.to_csv('data/valid.csv', index=False)


if __name__ == "__main__":
    download_data('https://storage.googleapis.com/vakyaansh-open-models/lid_data/lid_data.zip')
    explore_audio_data()