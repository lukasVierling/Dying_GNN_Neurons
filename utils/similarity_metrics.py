import argparse
import yaml
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
from torch import linalg


def distance_to_mean( node_features_dict):
    """Calcualte the distance to mean metric over node embeddings

    Parameters
    ----------
    node_features: node feature during forward pass, should be layers x nodes x hid_dim

    Returns
    ----------
    distance_dict : dictionary with similarity metric for every layer
    """
    distances_dict = {}
    for layer, node_features in node_features_dict.items():
        node_features = node_features.cpu()
        mean_features = torch.mean(node_features, dim=0)
        mean_features = (mean_features.unsqueeze(0)).expand(node_features.shape)
        distance_to_mean = torch.linalg.norm(node_features - mean_features, dim=(0,1))
        distances_dict[layer] = distance_to_mean.item()
    return distances_dict


def dirichlet_energy(graph, node_features):
    """Calcualte the dirichlet energy over a graph, given the node features for every layer

    Parameters
    ----------
    gaph: the dataset
    node_features: node feature during forward pass, should be layers x nodes x hid_dim

    Returns
    ----------
    energy : dictionary with similarity metric for every layer
    """
    layerwise_energy = {}
    for layer, activations in node_features.items():
      #layer is NxD
      edge_index = graph.edge_index
      num_nodes = activations.shape[0]  # num nodes = N

      energy = 0
      for i, j in zip(edge_index[0], edge_index[1]):
          # Compute the squared difference of the node features for this edge
          diff = activations[i] - activations[j]
          #print("Dimension of activation vector: ", activations[i].shape)
          energy += torch.sum(diff ** 2)  # Squared difference
      #divide by nodes
      energy = energy/num_nodes
      #squre to get measure
      energy = torch.sqrt(energy)
      layerwise_energy[layer] = energy.item() # Return Dirichlet energy as a scalar value

    return layerwise_energy

def evaluate_similarity(
    model,
    data,
    metric="distance",
    random = True,
):
    """Calcualte the similarity for each layer, given a metric

    Parameters
    ----------
    model: the model we want to calc the similarity score for
    data: dataset for the forward
    metric: "distance" or "dirichlet", to decide which metric to use
    random: True or False, uses the original node features or random initialized node features

    Returns
    ----------
    result: is the evaluation of the metric over each layer
    """
    #data
    X = data.x
    Y = data.y
    A = data.edge_index
    graph = data
    if random:
        #compare to random input features
        data = torch.randn_like(X)
    else:
        data = X
    activations = model.generate_node_embeddings(data,A)
    activations = {i: act.clone().detach().cpu() for i,act in enumerate(activations)}
    if metric=="dirichlet":
        result = dirichlet_energy(graph,activations)
    elif metric=="distance":
        result = distance_to_mean(activations)
    else:
        print("undefined metric!")
        result = {}
    return result