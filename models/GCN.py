import torch
import os
import typing
import torch_geometric

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

class GCN(nn.Module):
  def __init__(
      self,
      input_dim: int,
      hid_dim: int,
      n_classes: int,
      n_layers: int,
      activation: str):
    super(GCN, self).__init__()
    """
    Args:
      input_dim: input feature dimension
      hid_dim: hidden feature dimension
      n_classes: number of target classes
      n_layers: number of layers
      dropout_ratio: dropout_ratio
    """
    ## ------ Begin Solution ------ ##
    self.input_dim = input_dim
    self.hid_dim = hid_dim
    self.n_classes = n_classes
    self.n_layers = n_layers
    self.layers = None
    if activation == "relu":
      self.activation = F.relu
    elif activation == "leakyrelu":
      self.activation = nn.LeakyReLU(0.8)
    else:
      print("Undefined activation function!")
    layers = []
    #start stacking the layers
    for i in range(self.n_layers):
      layers.append(GCNConv(self.hid_dim if i != 0 else self.input_dim, self.hid_dim))
    layers.append(nn.Linear(self.hid_dim if self.n_layers != 0 else self.input_dim, self.n_classes))

    self.layers = nn.ModuleList(layers)


    ## ------ End Solution ------ ##

  def forward(self, X, A) -> torch.Tensor:
    ## ------ Begin Solution ------ ##
    for layer in self.layers:

      if not isinstance(layer, nn.Linear):
        # no relu and dropout on last layer
        X = layer(X,A)
        X = self.activation(X)
      else:
        X = layer(X)
    return X
    ## ------ End Solution ------ ##

  def generate_node_embeddings(self, X, A) -> torch.Tensor:
    ## ------ Begin Solution ------ ##
    embeddings = []
    #skip the forward through the linear layer so no logits
    for layer in self.layers[:-1]:
      if not isinstance(layer, nn.Linear):
        # no relu and dropout on last layers
        X = layer(X,A)
        X = self.activation(X)
        #X = F.dropout(X,self.dropout_ratio)
        embeddings.append(X)
      else:
        X = layer(X)
        embeddings.append(X)
    return embeddings
    ## ------ End Solution ------ ##
  def param_init(self):
    """
    Initialize the parameters of the model to prevent dying neurons.
    """
    for layer in self.layers:
        if isinstance(layer, nn.Linear):
            # He initialization for Linear layers
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
