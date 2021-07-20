from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
from parameters import SAMPLE_RATE, FEATURE_EXTRACTION, DATA_FILE, CROP_DURATION

class LanguageIdentificationDataset(Dataset):

    def __init__(self):

        self.data_file = pd.read_csv(DATA_FILE)
        self.feature_extraction = FEATURE_EXTRACTION
        self.crop_duration = CROP_DURATION
        self.sample_rate = SAMPLE_RATE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def __len__(self):
        return len(self.data_file)

    def get_audio_path(self, idx):
        return self.data_file['path'][idx]
    
    def get_audio_language_label(self, idx):
        return self.data_file['label'][idx]
    
    def audio_transform(self, raw_audio):

        if self.feature_extraction == 'raw audio':
            return raw_audio

        elif self.feature_extraction == 'mel spectrogram':
            return torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )(raw_audio)

        else:
            print("Incorrect feature extraction method")

    
    def crop_or_pad_audio(self, raw_audio):

        num_frames = self.crop_duration * self.sample_rate

        if raw_audio.shape[1] >= num_frames:
            raw_audio = raw_audio[:, :num_frames]
            return raw_audio

        else:
            padding_dimension = (0, num_frames - raw_audio.shape[1])
            return torch.nn.functional.pad(raw_audio, padding_dimension)


    def __getitem__(self, idx):
        audio_path = self.get_audio_path(idx)
        language_label = self.get_audio_language_label(idx)

        audio_signal, _ = torchaudio.load(audio_path)
        audio_signal = audio_signal.to(self.device)
        language_label = torch.tensor(int(language_label)).to(self.device)
        audio_signal = self.crop_or_pad_audio(audio_signal)
        audio_signal = self.audio_transform(audio_signal)

        return audio_signal, language_label


if __name__ == "__main__":
    lid = LanguageIdentificationDataset()
    s, l = lid[0]

    print(s,l)
    print(type(s))
    print(type(l))




    