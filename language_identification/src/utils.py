import librosa
import numpy as np


def load_wav(audio_filepath, sr, min_dur_sec=5):
    audio_data, fs = librosa.load(audio_filepath, sr=16000)
    len_file = len(audio_data)

    if len_file <= int(min_dur_sec * sr):
        temp = np.zeros((1, int(min_dur_sec * sr) - len_file))
        joined_wav = np.concatenate((audio_data, temp[0]))
    else:
        joined_wav = audio_data
    return joined_wav


def spectogram_from_wav(wav, hop_length, win_length, n_fft=512):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    return linear.T


def load_data(filepath, sr=16000, min_dur_sec=5, win_length=400, hop_length=160, n_mels=40, spec_len=400):
    audio_data = load_wav(filepath, sr=sr, min_dur_sec=min_dur_sec)
    linear_spect = spectogram_from_wav(audio_data, hop_length, win_length, n_fft=512)
    mag, _ = librosa.magphase(linear_spect)
    mag = np.log1p(mag)
    mag_T = mag.T


    randtime = np.random.randint(0, mag_T.shape[1] - spec_len)
    spec_mag = mag_T[:, randtime:randtime + spec_len]


    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)