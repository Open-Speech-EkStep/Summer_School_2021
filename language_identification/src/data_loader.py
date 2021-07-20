from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class LanguageIdentificationDataset(Dataset):

    def __init__(self, data_file):

        self.data_file = pd.read_csv(data_file)


    def __len__(self):
        return len(self.data_file)

    def get_audio_path(self, idx):
        return self.data_file['path'][idx]
    
    def get_audio_language_label(self, idx):
        return self.data_file['label'][idx]

    def __getitem__(self, idx):
        audio_path = self.get_audio_path(idx)
        language_label = self.get_audio_language_label(idx)

        raw_audio, sr = torchaudio.load(audio_path)

        return raw_audio, language_label

if __name__ == "__main__":
    lid = LanguageIdentificationDataset('data/audio_data.csv')
    print(len(lid))

    s, l = lid[0]
    print(s)
    print(l)




    