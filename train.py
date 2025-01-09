import torch
import os
import typing
import torch_geometric
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.datasets as datasets
import random
from tqdm import tqdm

from torch_geometric.nn import GCNConv

import time
import copy

from monitor.monitor import monitor
from models.GCN import GCN



def train(
    dataset,
    params: typing.Dict,
    dead_threshold,
    prune = True,
    reinit = "0",
    prune_every_n = 1,
    prune_warmup = 0,

) -> torch.nn.Module:
    """Trains a GNN.

    Parameters
    ----------
    dataset: data
    params: a dictionary with parameters required for the training and model setup
    dead_threshold: threshold for when a neuron is dead
    prune: bool to decide if prune the model or not
    reinit: strategy how to reinit the model
    prune_every_n : prune the model every n epochs
    prune_warmup: don't prune for the first k epochs

    Returns
    ----------
    dead_neurons_tracker : a dict over layers, amount of dead neurons per layer
    val_accuracies : val accuracies calculated every time we prune (or not prune) (default every epoch)
    train_accuracies : training accuarcies for every epoch
    test_acc : test accuracy of the best model
    test_dead_neurons_averaged : dead neurons in the whole model for best model during forward on test data
    best_model : the model with the highest val accuracy 
    """

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load dataset
    data = dataset.data
    data = data.to(device)

    params["n_classes"] = dataset.num_classes # number of target classes
    params["input_dim"] = dataset.num_features # size of input features

    epochs = params["epochs"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    
    model = GCN(
    params["input_dim"],
    params["hid_dim"],
    params["n_classes"],
    params["n_layers"],
    params["activation"],
    ).to(device)
  
    model.param_init()

    train_mask =  dataset.data.train_mask.bool()
    val_mask = dataset.data.val_mask.bool()
    test_mask = dataset.data.test_mask.bool()

    X = dataset.data.x
    Y = dataset.data.y
    A = dataset.data.edge_index

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # define loss function
    loss_func = torch.nn.CrossEntropyLoss()
    # start training, setup model
    model.to(device)
    model.train()

    #logging stuff
    losses = []

    dead_neurons_tracker = {}
    val_accuracies = {}
    train_accuracies = {}
    best_acc = 0
    best_model = None

    for epoch in range(epochs):
        # forward the input
        out = model(X,A)

        # calc the loss over the train x and y
        out_train = out[train_mask]
        Y_train = Y[train_mask]#calc the train acc
        with torch.no_grad():  
                predicted = torch.argmax(out_train, dim=1)  # Get predicted class
                train_correct = (predicted == Y_train).sum().item()  # Count correct predictions as a scalar (on CPU)
                train_total = train_mask.sum().item()  # Total number of training samples
                train_acc = train_correct / train_total  # Calculate accuracy as a scalar number

        train_accuracies[epoch] = train_acc

        #calc loss and grad update
        loss = loss_func(out_train,Y_train)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # start montioring 
        if epoch > prune_warmup and not (epoch % prune_every_n):  # Evaluate every epoch
            model.eval()
            temp=copy.deepcopy(model) #save model before pruning in case its best performing
            val_acc, dead_neurons = monitor(
                    model, data, val_mask, dead_threshold, prune, reinit=reinit
            ) #prune is passed as argument decides if weights are pruned or not
            model.train()

            dead_neurons_tracker[epoch] = dead_neurons

            val_accuracies[epoch] = val_acc

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = temp

    # evaluate on the testset
    model.eval()
    test_acc, test_dead_neurons = monitor(
        model, data, test_mask, dead_threshold, prune=False
    )
    test_dead_neurons_averaged = np.mean(np.array(test_dead_neurons))
    model.train()

    # return all metrics and the best model
    return dead_neurons_tracker, val_accuracies, train_accuracies, test_acc, test_dead_neurons_averaged, best_model
