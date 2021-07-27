import os
import sys
import numpy as np
import torch
import yaml
import warnings
warnings.filterwarnings("ignore")
from utils import load_data
from model import get_model

if len(sys.argv) < 3:
    print('Usage: python infer.py [checkpoint path] [audio path]')
    exit()

language_map = {'0': 'Gujarati', '1': 'Kannada', '2': 'Telugu'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(device, 3, pretrained=False)

checkpoints = torch.load(sys.argv[1])

model.load_state_dict(checkpoints['state_dict'])

model.eval()

spec = load_data(sys.argv[2])[np.newaxis, ...]
feats = np.asarray(spec)
feats = torch.from_numpy(feats)
feats = feats.unsqueeze(0)
feats = feats.to(device)
label = model(feats.float())
_, prediction =label.max(1)

print(language_map[str(prediction.cpu().detach().numpy()[0])])
