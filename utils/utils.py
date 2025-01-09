
import argparse
import yaml
import torch
import os
import typing
import torch_geometric
import numpy as np

import json

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.datasets as datasets
import random
from tqdm import tqdm

from torch_geometric.nn import GCNConv

import time
import copy

from collections import defaultdict

def parse_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def average_dict(dict,n):
    dict["dead_neurons"] = [dead_neurons / n for dead_neurons in dict["dead_neurons"]]
    dict["val_accuracies"] = [val_acc / n for val_acc in dict["val_accuracies"]]
    dict["train_accuracies"] = [train_acc / n for train_acc in dict["train_accuracies"] ]
    dict["test_acc"] = dict["test_acc"] / n
    dict["test_dead_neurons"] = dict["test_dead_neurons"]/n
    return dict

def update_dict(dict, dead_neurons_tracker, val_accuracies, train_accuracies, test_acc, test_dead_neurons):
    dict["dead_neurons"] = [dead_neurons_prev + np.mean(dead_neurons) for dead_neurons_prev, dead_neurons in zip(dict["dead_neurons"], dead_neurons_tracker.values())]
    dict["val_accuracies"] = [prev_val_acc + val_acc for prev_val_acc, val_acc in zip(dict["val_accuracies"], val_accuracies.values())]
    dict["train_accuracies"] = [prev_train_acc + train_acc for prev_train_acc, train_acc in zip(dict["train_accuracies"], train_accuracies.values())]
    dict["test_acc"] = dict["test_acc"] + test_acc
    dict["test_dead_neurons"] = dict["test_dead_neurons"] + test_dead_neurons
    return dict

def init_stats_dict(epochs):

    return {
        "dead_neurons": [0]*epochs, 
        "val_accuracies": [0]*epochs,   
        "train_accuracies": [0]*epochs,     
        "test_acc": 0,         
        "test_dead_neurons": 0, 
    }

def save_best_models(models_dict, results_path, model_str, experiment_id):

    for n_layers, best_models in models_dict.items():

        for idx, model in enumerate(best_models):
            model_path = os.path.join(
                results_path, 
                f"{model_str}_best_model_layers_{n_layers}_run_{idx}_{experiment_id}.pth"
            )
            torch.save(model.state_dict(), model_path)