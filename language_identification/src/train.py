import os
import torch.nn as nn
from torch.nn import parameter
import torch.optim as optim
import torch
import parameters
from data_loader import LanguageIdentificationDataset
from torchvision.models import resnet18
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = LanguageIdentificationDataset(parameters.TRAIN_DATA_FILE)
valid_dataset = LanguageIdentificationDataset(parameters.VALID_DATA_FILE)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters.BATCH_SIZE, num_workers=8)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=parameters.BATCH_SIZE, num_workers=8)

model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 4)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.to(device=device)

optimizer = optim.Adam(model.parameters(), lr=parameters.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

starting_epoch = 0

if parameters.LOAD_CHECKPOINT:
    checkpoint = torch.load(parameters.CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

loss_values, validation_loss_values = [], []

train_step_count = 0

os.makedirs(parameters.CHECKPOINT_DIR, exist_ok=True)

for epoch in range(starting_epoch, parameters.EPOCHS):

    model.train()
    total_loss = 0

    tk0 = tqdm(train_data_loader, total=int(len(train_data_loader)), unit='batch')
    tk0.set_description(f'Epoch {epoch + 1}')

    for step, batch in enumerate(tk0):

        input = batch[0].to(device)
        target = batch[1].to(device)

        model.zero_grad()

        output = model(input)

        loss = criterion(output, target)

        loss.backward()
        total_loss += loss

        train_step_count += 1

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=parameters.MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_data_loader)
    print("Average train loss: {}".format(avg_train_loss))
    print("total loss", total_loss)

    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}

    torch.save(state, parameters.CHECKPOINT_DIR + '/checkpoint_last.pt')

    loss_values.append(avg_train_loss)

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_labels, true_labels = [], []

    best_val_loss = np.inf 

    for batch in tqdm(valid_data_loader, total=int(len(valid_data_loader)), unit='batch', leave=True):

        with torch.no_grad():
            output = model(batch[0].to(device))

        label = batch[1].to(device)

        eval_loss += criterion(output, label)
        _, predictions = output.max(1)
        predictions = predictions.detach().cpu().numpy()
        eval_labels.extend([p for p in list(predictions)])
        true_labels.extend(label.detach().cpu().numpy())

    eval_loss = eval_loss / len(valid_data_loader)

    if eval_loss < best_val_loss:
        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, parameters.CHECKPOINT_DIR + '/checkpoint_best.pt')
        best_val_loss = eval_loss

    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    #true_labels = true_labels.detach().cpu().numpy()
    val_accuracy = accuracy_score(true_labels, eval_labels)
    val_f1_score = f1_score(true_labels, eval_labels, average='macro')

    #print(val_accuracy)
    #print(val_f1_score)
    print("Validation Accuracy: {}".format(val_accuracy))
    print("Validation F1-Score: {}".format(val_f1_score))
    #print("Classification Report: {}".format(classification_report(true_labels, eval_labels, output_dict=True,
     #                                                              labels=np.unique(eval_labels))))