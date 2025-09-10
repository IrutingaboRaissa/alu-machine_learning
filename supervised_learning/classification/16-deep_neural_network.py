#!/usr/bin/env python3
"""
Module containing the DeepNeuralNetwork class for binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    A class that defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Initialize a DeepNeuralNetwork instance

        Args:
            nx (int): The number of input features
            layers (list): List representing the number of nodes in each layer

        Raises:
            TypeError: If nx is not an integer or layers is not a list of positive integers
            ValueError: If nx is not positive
        """
        # Validate nx parameter
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Validate layers parameter
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        
        # Check if all elements in layers are positive integers
        for layer_size in layers:
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")
        
        # Set public instance attributes
        self.L = len(layers)  # Number of layers in the neural network
        self.cache = {}  # Dictionary to hold all intermediary values
        self.weights = {}  # Dictionary to hold all weights and biases
        
        # Initialize weights and biases for each layer
        prev_layer_size = nx  # Start with input size
        
        for layer_index in range(self.L):
            current_layer_size = layers[layer_index]
            layer_num = layer_index + 1  # Layer numbering starts from 1
            
            # Initialize weights using He et al. method
            # He initialization: W = np.random.randn(layer_size, prev_layer_size) * sqrt(2 / prev_layer_size)
            self.weights[f"W{layer_num}"] = np.random.randn(current_layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            
            # Initialize biases to zeros
            self.weights[f"b{layer_num}"] = np.zeros((current_layer_size, 1))
            
            # Update prev_layer_size for next iteration
            prev_layer_size = current_layer_size
