import os
import shutil

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import SpeechDataGenerator

def create_output_dirs(checkpoint_path):
    os.makedirs(os.path.join(checkpoint_path, "current_checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_path, "best_model"), exist_ok=True)
    os.makedirs(os.path.join(checkpoint_path, "final_model"), exist_ok=True)


def load_yaml_file(path):
    read_dict = {}
    with open(path, 'r') as file:
        read_dict = yaml.safe_load(file)
    return read_dict


# Load Data
def load_data_loaders(train_manifest, valid_manifest, batch_size):
    train_data = SpeechDataGenerator(manifest=train_manifest, mode='train')
    test_data = SpeechDataGenerator(manifest=valid_manifest, mode='train')

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())

    loaders = {
        'train': train_loader,
        'test': test_loader,
    }
    return loaders


def show_model_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Total paramaeters: ", sum([np.prod(p.size()) for p in model.parameters()]), "\nTrainable parameters: ",
          params)


def save_ckp(state, model, valid_loss, valid_loss_min, checkpoint_path, best_model_path, final_model_path,
             save_for_each_epoch):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    if save_for_each_epoch:
        checkpoint_name = final_model_path.split("/")[-1]
        new_checkpoint_name = str(state['epoch']-1) + "_" + checkpoint_name
        f_path = os.path.join(final_model_path.replace(checkpoint_name, new_checkpoint_name))
        torch.save(model, f_path)

    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if valid_loss <= valid_loss_min:
        best_fpath = best_model_path
        torch.save(model, final_model_path)
        shutil.copyfile(f_path, best_fpath)


def train(start_epochs, n_epochs, device, valid_loss_min_input, loaders, model, optimizer, criterion, use_cuda,
          checkpoint_path, save_for_each_epoch=True):
    """
    Keyword arguments:
    start_epochs -- the real part (default 0.0)
    n_epochs -- the imaginary part (default 0.0)
    valid_loss_min_input
    loaders
    model
    optimizer
    criterion
    use_cuda
    checkpoint_path
    best_model_path

    returns trained model
    """
    # initialize tracker for minimum validation loss
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=1e-1, patience=1, verbose=True)
    valid_loss_min = valid_loss_min_input

    # create checkpoints
    path = checkpoint_path
    checkpoint_path = os.path.join(path, "current_checkpoint", "current_checkpoint.pt")
    best_model_path = os.path.join(path, "best_model", "best_checkpoint.pt")
    final_model_path = os.path.join(path, "final_model", "final_model.pt")

    # load checkpoint from last run if available
    if os.path.isfile(checkpoint_path):
        print("loaded model from ", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epochs = checkpoint['epoch']
        valid_loss_min = checkpoint['valid_loss_min']

    create_output_dirs(path)

    for epoch in range(start_epochs, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_predict = []
        valid_predict = []
        train_target = []
        valid_target = []
        temp_predict = []
        temp_target = []
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in tqdm(enumerate(loaders['train']), total=len(loaders['train']), leave=False):
            # move to GPU
            data, target = data.to(device, dtype=torch.float), target.to(device)
            ## find the loss and update the model parameters accordingly
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            ## record the average training loss, using something like
            _, predictions = output.max(1)
            temp_predict = [pred.item() for pred in predictions]
            temp_target = [actual.item() for actual in target]

            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

            train_predict = train_predict + temp_predict
            train_target = train_target + temp_target

        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in tqdm(enumerate(loaders['test']), total=len(loaders['test']), leave=False):
            # move to GPU
            if use_cuda:
                data, target = data.to(device, dtype=torch.float), target.to(device)
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            _, predictions = output.max(1)
            temp_predict = [pred.item() for pred in predictions]
            temp_target = [actual.item() for actual in target]

            valid_predict = valid_predict + temp_predict
            valid_target = valid_target + temp_target

        # calculate average losses
        train_loss = train_loss / len(loaders['train'])
        valid_loss = valid_loss / len(loaders['test'])
        train_acc = accuracy_score(train_target, train_predict)
        valid_acc = accuracy_score(valid_target, valid_predict)

        # print training/validation statistics
        print(
            'Epoch: {} \tTraining Loss: {:.10f} \tTraining Accuracy: {:.6f} \tValidation Loss: {:.10f} \tValidation  Accuracy: {:.6f} '.format(
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc
            ))

        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        scheduler.step(valid_loss)
        # save checkpoint
        save_ckp(checkpoint, model, valid_loss, valid_loss_min, checkpoint_path, best_model_path, final_model_path,
                 save_for_each_epoch)

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).    Model Saved......'.format(valid_loss_min, valid_loss))
            # save_ckp(checkpoint, model, True, checkpoint_path, best_model_path, final_model_path)
            valid_loss_min = valid_loss
    return model