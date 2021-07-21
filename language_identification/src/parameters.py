SAMPLE_RATE = 16000
FEATURE_EXTRACTION = 'mel spectrogram' # 'raw audio' 'mel spectrogram'
TRAIN_DATA_FILE = 'data/train.csv'
VALID_DATA_FILE = 'data/valid.csv'
CROP_DURATION = 2

# Training Parameters
CHECKPOINT_DIR = 'checkpoints'
BATCH_SIZE = 1024
LEARNING_RATE = 0.0005
EPOCHS = 10
LOAD_CHECKPOINT = False
CHECKPOINT_PATH = ''
MAX_GRAD_NORM = 1.0