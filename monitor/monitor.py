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

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def monitor(
    model,
    data,
    val_mask,
    dead_threshold=1,
    prune=False,
    k=10,
    reinit="0",
    prune_all=False,
):
    """Monitors the behaviour of the model.
    Evaluates the model on the val set and returns the accuracy.
    Tracks dead neurons during forward pass on val.
    Optional: prunes dead neurons.

    Parameters
    ----------
    model: the model
    data: the data
    val_mask: mask for the validation nodes in the dataset
    dead_threshold: percentage of dying relu to be considered dead e.g. 0.95 -> neurons that are 0 95% of the time are dead
    prune: if True then prune weights leading to dying relus
        only apply when pruning is turned on
        k: k/100 = percentage of weights pruned
        reinit: strategy to initialize the pruned weights
        prune_all: prune k/100% of weights wehn turned on, k/100% of negative weights otherwise
    """
    # Dictionary to store activations
    activations = {}

    # Source: https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
    def getActivation(name):
        def hook(model, input, output):
            # Apply ReLU on the activations (linear layer outputs)
            activations[name] = F.relu(output.detach().cpu())
        return hook

    # Register hook to all layers in the model
    for name, module in model.named_modules():
        if isinstance(module, GCNConv):
            module.register_forward_hook(getActivation(name))

    # Get the data
    X = data.x
    Y = data.y
    A = data.edge_index

    # Forward pass
    out = model(X, A)
    out = torch.argmax(out, dim=1)

    # Calculate validation accuracy
    out_val = out[val_mask]
    Y_val = Y[val_mask]
    correct = out_val == Y_val
    acc = torch.mean(correct.float()).item()
    #start pruning when pruning is set to true
    dead_neurons = []
    #print("Activations: ",activations)
    for name, outputs in activations.items():
        dead_fraction = np.mean(
            (np.mean((np.array(outputs) == 0), axis=0) >= dead_threshold)
        )
        dead_neurons.append(np.array(dead_fraction))

        # Identify dead neurons
        is_dead = np.mean((np.array(outputs) == 0), axis=0) >= dead_threshold

        if prune:
            if name in dict(model.named_modules()).keys():  # Ensure layer exists in the model
                layer = dict(model.named_modules())[name]  # Access the layer
                if isinstance(layer, GCNConv):  # Ensure it's the correct type of layer
                    if hasattr(layer, 'lin') and hasattr(layer.lin, 'weight'):  # Check for 'lin' module
                        with torch.no_grad():
                            weight = layer.lin.weight.clone().detach().cpu().numpy()  # Actually not creating a copy!
                            std = np.sqrt(np.var(weight))  # Calculate standard deviation of the weights
                            num_inputs = weight.shape[1]

                            # Loop through each neuron and adjust weights for dead ones
                            for idx, dead in enumerate(is_dead):
                                if dead:
                                    # Step 1: If prune_all is True, prune the most negative weights from all weights.
                                    if prune_all:
                                        num_weights_to_adjust = max(1, int(k / 100 * weight.shape[1]))  # Adjust this depending on your criterion
                                        most_negative_indices = np.argsort(weight[idx])[:num_weights_to_adjust]
                                    else:
                                        # Step 2: Sort the indices of negative weights by their values (most negative first)
                                        negative_weight_indices = np.where(weight[idx] < 0)[0]  # Get indices of negative weights

                                        # Ensure there are negative weights before proceeding
                                        if len(negative_weight_indices) > 0:
                                            sorted_indices = negative_weight_indices[np.argsort(weight[idx, negative_weight_indices])]

                                            # Step 3: Select the top 10% most negative weights
                                            num_weights_to_adjust = max(1, int(k/100 * len(sorted_indices)))  # Adjust the percentage as needed
                                            most_negative_indices = sorted_indices[:num_weights_to_adjust]
                                        else:
                                            # If no negative weights are found, ensure an empty pruning action
                                            most_negative_indices = []
                                            num_weights_to_adjust = 0

                                    # Apply the reinitialization strategy
                                    if reinit == "zero":
                                        weight[idx, most_negative_indices] = 0
                                    elif reinit == "normal":
                                        # Reinitialize weights from normal distribution around 0 with current  variance
                                        weight[idx, most_negative_indices] = np.random.normal(0, std, size=num_weights_to_adjust)
                                    elif reinit == "he":
                                        # Reinitialize weights using He Initialization
                                        weight[idx, most_negative_indices] = np.random.normal(0,np.sqrt(2 / num_inputs),size=num_weights_to_adjust)
                                    else:
                                        print("Undefined reinit strategy.")

                            layer.lin.weight.copy_(torch.tensor(weight, dtype=layer.lin.weight.dtype))

                else:
                    print(f"The layer {name} does not have a `lin` submodule with weights.")

    return acc, dead_neurons
