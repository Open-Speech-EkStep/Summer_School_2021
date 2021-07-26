import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np 
import wandb
from cnn import CNNNetwork

from model import get_model
from training_utils import *

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

train_parameters = load_yaml_file("train_config.yml")["train_parameters"]

# inputs
train_manifest = train_parameters["train_manifest"]
valid_manifest = train_parameters["valid_manifest"]
# outputs
checkpoint_path = train_parameters["checkpoint_path"]

# Hyperparameters
batch_size = int(train_parameters["batch_size"])
learning_rate = float(train_parameters["learning_rate"])
num_epochs = int(train_parameters["num_epochs"])
num_classes = int(train_parameters["num_classes"])

# Load_Data
loaders = load_data_loaders(train_manifest, valid_manifest, batch_size)

# Load Model
model = get_model(device, num_classes, pretrained=False)
# Set model hyperparameters
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()



config = {
  "model": "Resnet18",
  "learning_rate": learning_rate,
  "batch_size": batch_size,
  'num_classes': num_classes,
  'num_epochs': num_epochs,
  'learning_rate': learning_rate
}

wandb.init(project="test", config=config)
wandb.watch(model)

# start model trainning
trained_model = train(1, num_epochs, device, np.Inf, loaders, model, optimizer, criterion, use_cuda, checkpoint_path,
                      save_for_each_epoch=True)
