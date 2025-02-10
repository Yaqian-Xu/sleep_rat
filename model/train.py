
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import os.path as path
import pandas as pd

def get_accuracy(model, data_loader, criterion):
    pass

def train(model, train_loader, valid_loader, num_epochs=20, learning_rate=1e-3, plot_stat=0):
    pass
    # torch.manual_seed(42)
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train_acc = np.zeros(num_epochs)
    # val_acc = np.zeros(num_epochs)
    # max_importance_feat =  defaultdict(lambda: 0)
    # for epoch in range(num_epochs):
    #     total_train_acc = 0.0
    #     total_epoch=0
    #     for batch, (data, labels) in enumerate(train_loader):
    #         data = torch.autograd.Variable(data.float(), requires_grad=True)
    #         outputs = model(data).squeeze()
    #         loss = criterion(outputs, labels.float())
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         corr = (outputs > 0.0).squeeze().long() == labels

    #         gradients = data.grad.mean(dim=0)
    #         gradients /= torch.norm(gradients, p=1)
    #         feature_importance = torch.abs(gradients)
    #         feature_importance_np = feature_importance.detach().numpy()
    #         ind = np.argsort(feature_importance_np)[-3:][::-1]
    #         for i in ind:
    #           max_importance_feat[i] += 1

    #         total_train_acc += int(corr.sum())
    #         total_epoch += len(labels)
    #     train_acc[epoch] = float(total_train_acc)/total_epoch
    #     val_acc[epoch] = get_accuracy(model, valid_loader, criterion)
    #     print(("Epoch {}: Train acc: {}, Validation acc: {}").format(epoch + 1, train_acc[epoch], val_acc[epoch]))
    # if plot_stat:
    #   plt.title("Accuracy Curve at 10 minutes")
    #   plt.plot(range(num_epochs), train_acc, label="Train")
    #   plt.plot(range(num_epochs), val_acc, label="Validation")
    #   plt.xlabel("Epochs")
    #   plt.ylabel("Training Accuracy")
    #   plt.legend(loc='best')
    #   plt.show()
    # print("Final Training Accuracy: {}".format(train_acc[-1]))
    # print('Finished Training')
    # return max_importance_feat