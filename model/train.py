
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

from model import CNNRat
from dataset import *

def get_accuracy(model, data_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Validation", unit="batch") as pbar:
            for data, labels in data_loader:
                data, labels = data.to(device, dtype=torch.float32), labels.to(device)
                outputs = model(data.float())
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.update(1)
    accuracy = 100 * correct / total
    return accuracy, total_loss / len(data_loader)

def train(model, train_loader, valid_loader, num_epochs=20, learning_rate=5e-5, plot_results=False):
    # class weights
    class_counts = np.bincount([label for _, label in train_loader.dataset])
    class_weights = 1. / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    criterion = nn.CrossEntropyLoss(class_weights)
    train_acc = np.zeros(num_epochs)
    valid_acc = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for batch, (data, labels) in enumerate(train_loader):
                data, labels = data.to(device, dtype=torch.float32), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # this is to just update pbar
                pbar.set_postfix(loss=running_loss/(batch+1))
                pbar.update(1)
        
        train_acc[epoch], train_loss = get_accuracy(model, train_loader, criterion)
        valid_acc[epoch], valid_loss = get_accuracy(model, valid_loader, criterion)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_acc[epoch]:.2f}%, "
              f"Validation Loss: {valid_loss:.4f}, "
              f"Validation Accuracy: {valid_acc[epoch]:.2f}%")

    if (plot_results):
        plt.title(f"Accuracy for {num_epochs} epochs")
        plt.plot(range(num_epochs), train_acc, label="Train")
        plt.plot(range(num_epochs), valid_acc, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.show()

root_dir = os.getcwd()
edf_dirA = root_dir + '/training_data/CohortA/recordings/'
edf_dirB = root_dir + '/training_data/CohortB/recordings/'
edf_dirC = root_dir + '/training_data/CohortC/recordings/'
edf_dirD = root_dir + '/training_data/CohortD/recordings/'
label_dirA = root_dir + '/training_data/CohortA/scorings/'
label_dirB = root_dir + '/training_data/CohortB/scorings/'
label_dirC = root_dir + '/training_data/CohortC/scorings/'
label_dirD = root_dir + '/training_data/CohortD/scorings/'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train/test SleepyRat model")
    parser.add_argument('--save_model', type=str, help="Save model to dir path provided")
    parser.add_argument('--test_model', type=str, help="Test the model (.pth) saved at path provided")

    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.backends.mps.allow_tf32 = True
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # SleepRat paper outlined training was only done on Cohort A, rest of data was used for testing + validation
    train_dirs = [edf_dirA]
    train_label_dirs = [label_dirA]
    valid_dirs = [edf_dirB, edf_dirC]
    valid_label_dirs = [label_dirB, label_dirC]
    test_dirs = [edf_dirD]
    test_label_dirs = [label_dirD]
    batch_size = 64
    num_epochs = 5

    train_dataset = EEGDataset(train_dirs, train_label_dirs)
    test_dataset = EEGDataset(test_dirs, test_label_dirs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    model = CNNRat().to(device)
    if args.test_model: 
        model.load_state_dict(torch.load(args.test_model))
        test_acc, _ = get_accuracy(model, test_loader)
        print(f"Testing accuracy: {test_acc:.2f}%")
    else: 
        valid_dataset = EEGDataset(valid_dirs, valid_label_dirs)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)
        train(model, train_loader, valid_loader, num_epochs=num_epochs)

        test_acc, _ = get_accuracy(model, test_loader)
        print(f"Testing accuracy on Cohort D: {test_acc:.2f}%")
        
        if args.save_model:
            # save_model_path = f"{root_dir}/model/"
            save_model_path = args.save_model
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path, exist_ok=True)
            torch.save(model.state_dict(), f"{save_model_path}/model_{num_epochs}e.pth")
