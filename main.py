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

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from train import train
from utils.similarity_metrics import evaluate_similarity
from utils.utils import update_dict,average_dict,init_stats_dict,parse_config,save_best_models

RELU_COLOR = '#FFA500'
PRUNING_COLOR = '#87CEEB' 
LEAKY_COLOR = '#98FB98' 

def main():
    ## parse arguments
    parser = argparse.ArgumentParser(description="Parser for Training")
    parser.add_argument("--config", default="configs/base.yaml", type=str, help="Model and Training Config File")
    parser.add_argument("--experiment", default="train", type=str, help="Model and Training Config File")
    args = parser.parse_args()

    config = parse_config(args.config)

    training_params = config["training_params"]
    data_params = config["data_params"]
    prune_params = config["prune_params"]

    experiment = args.experiment

    if experiment == "train":
        print("Training starts with: ")
        print("Training Parameters: ", training_params)
        print("Dataset Parameters: ", data_params)

        dataset = datasets.Planetoid(*data_params.values())

        dead_neurons_tracker, val_accuracies, train_accuracies, test_acc, test_dead_neurons, best_model = train(dataset, training_params, *prune_params.values())
        print("Finished Training")
        print("Val Accuracies: ", val_accuracies)

    if experiment =="1":
        experiment_params = config["experiment_params"]
        layers = experiment_params["layers"]
        runs = experiment_params["runs"]
        results_path = experiment_params["results_path"]
        os.makedirs(results_path,exist_ok=True)
        experiment_id = experiment_params["id"] + "_" + str(time.time())
        print(f"Experiment 1: We will train models with layer depth {layers} for ReLU, ReLU with pruning, and LeakyReLU.")
        print(f"Each model will be trained {runs} times. This could take a long time.")

        dataset = datasets.Planetoid(*data_params.values())

        relu_stats_layer = {}
        relu_pruning_stats_layer = {}
        leaky_relu_stats_layer = {}

        relu_models_layers = {}
        relu_pruning_models_layers = {}
        leaky_relu_models_layers = {}
        epochs = training_params["epochs"]
        for n_layers in layers:
            relu_stats = init_stats_dict(epochs)
            relu_pruning_stats = init_stats_dict(epochs)
            leaky_relu_stats = init_stats_dict(epochs)
            relu_models = []
            relu_pruning_models = []
            leaky_relu_models = []

            training_params["n_layers"] = n_layers

            # Train ReLU <runs> times and collect results
            # No pruning
            training_params["activation"] = "relu"
            prune_params["prune"] = False
            for i in range(runs):
                dead_neurons_tracker, val_accuracies, train_accuracies, test_acc, test_dead_neurons, best_model = train(dataset, training_params, *prune_params.values())
                # Store the values
                # Dead neurons should be a list of length Epochs x L, so we take mean over all layers for every epcoh
                relu_stats = update_dict(relu_stats, dead_neurons_tracker, val_accuracies, train_accuracies, test_acc, test_dead_neurons)  
                relu_models.append(best_model)

            # Train relu + prune <runs> times and collect results
            # With pruning
            training_params["activation"] = "relu"
            prune_params["prune"] = True
            for i in range(runs):
                dead_neurons_tracker, val_accuracies, train_accuracies, test_acc, test_dead_neurons, best_model = train(dataset, training_params, *prune_params.values())
                # Store the values
                relu_pruning_stats = update_dict(relu_pruning_stats,dead_neurons_tracker, val_accuracies, train_accuracies, test_acc, test_dead_neurons)
                relu_pruning_models.append(best_model)

            # Train Leaky ReLU <runs> times and collect results
            training_params["activation"] = "leakyrelu"
            prune_params["prune"] = False
            for i in range(runs):
                dead_neurons_tracker, val_accuracies, train_accuracies, test_acc, test_dead_neurons, best_model = train(dataset, training_params, *prune_params.values())
                # Store the values
                leaky_relu_stats = update_dict(leaky_relu_stats,dead_neurons_tracker, val_accuracies, train_accuracies, test_acc, test_dead_neurons)
                leaky_relu_models.append(best_model)
            #Now we want to compute the statistics
            #Comppute averages
            relu_stats = average_dict(relu_stats,runs)
            relu_pruning_stats = average_dict(relu_pruning_stats,runs)
            leaky_relu_stats = average_dict(leaky_relu_stats, runs)
            #averages can be printes
            relu_stats_layer[n_layers] = relu_stats
            relu_pruning_stats_layer[n_layers] = relu_pruning_stats
            leaky_relu_stats_layer[n_layers] = leaky_relu_stats
            relu_models_layers[n_layers] = relu_models
            relu_pruning_models_layers[n_layers] = relu_pruning_models
            leaky_relu_models_layers[n_layers] = leaky_relu_models
        #start plotting the test dead neurons
        relu_dead_neurons = [relu_stats_layer[i]["test_dead_neurons"] for i in layers]
        relu_pruning_dead_neurons = [relu_pruning_stats_layer[i]["test_dead_neurons"] for i in layers]
        plt.plot(layers,relu_dead_neurons, color=RELU_COLOR)
        plt.plot(layers, relu_pruning_dead_neurons, color=PRUNING_COLOR)
        plt.xlabel("N Layers")
        plt.ylabel("Fraction Dead Neurons")
        plt.savefig(os.path.join(results_path,f"dead_neurons_{experiment_id}.png"), dpi=300)
        plt.close()
        #start plotting the test accuracies
        relu_test_accuracies = [relu_stats_layer[i]["test_acc"] for i in layers]
        relu_pruning_test_accuracies = [relu_pruning_stats_layer[i]["test_acc"] for i in layers]
        leaky_test_accuracies = [leaky_relu_stats_layer[i]["test_acc"] for i in layers]
        plt.plot(layers, relu_test_accuracies,label="ReLU", color=RELU_COLOR)
        plt.plot(layers, relu_pruning_test_accuracies,label="ReLU + Pruning", color=PRUNING_COLOR)
        plt.plot(layers, leaky_test_accuracies, label="LeakyReLU", color=LEAKY_COLOR)
        plt.xlabel("N Layers")
        plt.ylabel("Test Accuracy")
        plt.legend()
        plt.savefig(os.path.join(results_path,f"test_accuracies_plot_{experiment_id}.png"), dpi=300)
        plt.show()
        plt.close()
        #save the dictionaries in results folder
        #save the best models

        #save the dictionaries
        with open(os.path.join(results_path,"relu_stats_layer.json"), "w") as f:
            json.dump(relu_stats_layer, f)
        with open(os.path.join(results_path,"relu_pruning_stats_layer.json"), "w") as f:
            json.dump(relu_pruning_stats_layer, f)
        with open(os.path.join(results_path,"leaky_relu_stats_layer.json"), "w") as f:
            json.dump(leaky_relu_stats_layer, f)

        save_best_models(relu_models_layers,results_path, training_params["activation"]+"_"+training_params["model"]+"_"+str(prune_params["prune"]) ,experiment_id)
        save_best_models(relu_pruning_models_layers,results_path, training_params["activation"]+"_"+training_params["model"]+"_"+str(prune_params["prune"]) ,experiment_id)
        save_best_models(leaky_relu_models_layers,results_path, training_params["activation"]+"_"+training_params["model"]+"_"+str(prune_params["prune"]) ,experiment_id)
        
    if experiment == "2":
        experiment_params = config["experiment_params"]
        layers = experiment_params["layers"]
        runs = experiment_params["runs"]
        results_path = experiment_params["results_path"]
        #if not existing then create:
        os.makedirs(results_path,exist_ok=True)
        experiment_id = experiment_params["id"] + "_" + str(time.time())
        print(f"Experiment 2: We will train models with layer depth {layers} for ReLU, ReLU with pruning, and LeakyReLU.")
        print(f"Each model will be trained {runs} times. This could take a long time.")

        dataset = datasets.Planetoid(*data_params.values())
        
        epochs = training_params["epochs"]
        relu_similarities = { "dirichlet": [0] * layers, "distance":[0]*layers}
        relu_pruning_similarities= { "dirichlet": [0] * layers, "distance":[0]*layers}
        leaky_relu_similarities = { "dirichlet": [0] * layers, "distance":[0]*layers}

        #layers should be a single value
        training_params["n_layers"] = layers

        # Train ReLU <runs> times and collect results
        # No pruning
        training_params["activation"] = "relu"
        prune_params["prune"] = False
        for i in range(runs):
            _, _, _, _, _, best_model = train(dataset, training_params, *prune_params.values())
            dirichlet = evaluate_similarity(best_model, dataset, "dirichlet", random=False)
            distance = evaluate_similarity(best_model, dataset, "distance", random=False)
            relu_similarities["dirichlet"] = [prev_sim + sim for prev_sim,sim in zip(dirichlet.values(),relu_similarities["dirichlet"])]
            relu_similarities["distance"] = [prev_sim + sim for prev_sim,sim in zip(distance.values(),relu_similarities["distance"])]
        #average
        relu_similarities["dirichlet"] = [sim/runs for sim in relu_similarities["dirichlet"]]
        relu_similarities["distance"] = [sim/runs for sim in relu_similarities["distance"]]
        

        # Train ReLU <runs> times and collect results
        # With pruning
        training_params["activation"] = "relu"
        prune_params["prune"] = True
        for i in range(runs):
            _, _, _, _, _, best_model = train(dataset, training_params, *prune_params.values())
            dirichlet = evaluate_similarity(best_model, dataset, "dirichlet", random=False)
            distance = evaluate_similarity(best_model, dataset, "distance", random=False)
            relu_pruning_similarities["dirichlet"] = [prev_sim + sim for prev_sim,sim in zip(dirichlet.values(),relu_pruning_similarities["dirichlet"])]
            relu_pruning_similarities["distance"] = [prev_sim + sim for prev_sim,sim in zip(distance.values(),relu_pruning_similarities["distance"])]
            # Store the values
        relu_pruning_similarities["dirichlet"] = [sim/runs for sim in relu_pruning_similarities["dirichlet"]]
        relu_pruning_similarities["distance"] = [sim/runs for sim in relu_pruning_similarities["distance"]]

        # Train Leaky ReLU <runs> times and collect results
        training_params["activation"] = "leakyrelu"
        prune_params["prune"] = False
        for i in range(runs):
            _, _, _, _, _, best_model = train(dataset, training_params, *prune_params.values())
            dirichlet = evaluate_similarity(best_model, dataset, "dirichlet", random=False)
            distance = evaluate_similarity(best_model, dataset, "distance", random=False)
            leaky_relu_similarities["dirichlet"] = [prev_sim + sim for prev_sim,sim in zip(dirichlet.values(),leaky_relu_similarities["dirichlet"])]
            leaky_relu_similarities["distance"] = [prev_sim + sim for prev_sim,sim in zip(distance.values(),leaky_relu_similarities["distance"])]
            # Store the values
        leaky_relu_similarities["dirichlet"] = [sim/runs for sim in leaky_relu_similarities["dirichlet"]]
        leaky_relu_similarities["distance"] = [sim/runs for sim in leaky_relu_similarities["distance"]]
        #plot the similarities over layers
        print(relu_similarities)
        plt.plot(relu_similarities["dirichlet"],label="ReLU Dirichlet", color=RELU_COLOR)
        plt.plot(relu_pruning_similarities["dirichlet"],label="ReLU + Pruning Dirichlet", color=PRUNING_COLOR)
        plt.plot(leaky_relu_similarities["dirichlet"], label="LeakyReLU Dirichlet", color=LEAKY_COLOR)
        plt.xlabel("N Layers")
        plt.xscale("log")
        plt.ylabel("Similarity Metric")
        plt.yscale("log")
        plt.plot(relu_similarities["distance"],label="ReLU DTM", linestyle="--",color=RELU_COLOR)
        plt.plot(relu_pruning_similarities["distance"],label="ReLU + Pruning DTM", linestyle="--",color=PRUNING_COLOR)
        plt.plot(leaky_relu_similarities["distance"], label="LeakyReLU DTM",linestyle="--", color=LEAKY_COLOR)
        plt.legend()
        plt.savefig(os.path.join(results_path,f"similarities_plot_{experiment_id}.png"), dpi=300)
        plt.show()
        plt.close()

        #save the dictionaries
        with open(os.path.join(results_path,f"relu_similarities_{experiment_id}.json"), "w") as f:
            json.dump(relu_similarities, f)
        with open(os.path.join(results_path,f"relu_pruning_similarities_{experiment_id}.json"), "w") as f:
            json.dump(relu_pruning_similarities, f)
        with open(os.path.join(results_path,f"leaky_relu_similarities_{experiment_id}.json"), "w") as f:
            json.dump(leaky_relu_similarities, f)
        

if __name__ == "__main__":
    main()