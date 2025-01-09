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


def distance_to_mean(graph, node_features_dict):
  distances_dict = {}
  for layer, node_features in node_features_dict.items():
    node_features = node_features.cpu()
    mean_features = torch.mean(node_features, dim=0)
    mean_features = (mean_features.unsqueeze(0)).expand(node_features.shape)
    #print(mean_features.shape)
    distance_to_mean = torch.linalg.norm(node_features - mean_features, dim=(0,1))
    #print(distance_to_mean.shape)
    distances_dict[layer] = distance_to_mean.item()
  return distances_dict


def dirichlet_energy(graph, node_features):
    """
    Calculate the Dirichlet energy over all nodes, based on the given node features.

    :param graph: The dataset graph, which includes edge_index (sparse adjacency matrix).
    :param node_features: The feature matrix for all nodes (size num_nodes x feature_dim).
    :return: Dirichlet energy value for the entire dataset.
    """
    layerwise_energy = {}
    for layer, activations in node_features.items():
      #layer is NxD
      edge_index = graph.edge_index
      num_nodes = activations.shape[0]  # num nodes = N

      energy = 0

      # Loop through edges and compute squared differences between node features
      #print(energy)
      #print(f"doing {len(list((zip(edge_index[0], edge_index[1]))))} iterations")
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
    #data
    X = data.x
    Y = data.y
    A = data.edge_index
    graph = data
    #out = model(X,A)
    #calc the energy
    if random:
        #compare to random input features
        data = torch.randn_like(X)
    else:
        data = X
    activations = model.generate_node_embeddings(data,A)
    activations = {i: act.clone().detach().cpu() for i,act in enumerate(activations)}
    #print(activations)
    if metric=="dirichlet":
        energy = dirichlet_energy(graph,activations)
    elif metric=="distance":
        energy = distance_to_mean(graph,activations)
    else:
        print("undefined metric!")
        energy = {}
    #plot_dirichlet_energy(energy)
    return energy