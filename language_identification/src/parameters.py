SAMPLE_RATE = 16000
FEATURE_EXTRACTION = 'librosa' # 'power spectrogram' 'mel spectrogram'
TRAIN_DATA_FILE = 'data/train.csv'
VALID_DATA_FILE = 'data/valid.csv'
CROP_DURATION = 7

# Training Parameters
CHECKPOINT_DIR = 'checkpoints'
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
LOAD_CHECKPOINT = False
CHECKPOINT_PATH = ''
MAX_GRAD_NORM = 1.0