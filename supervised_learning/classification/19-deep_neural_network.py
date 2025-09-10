#!/usr/bin/env python3
"""
Module containing the DeepNeuralNetwork class for binary classification with cost calculation
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
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        
        # Check if all elements in layers are positive integers
        for layer_size in layers:
            if not isinstance(layer_size, int) or layer_size <= 0:
                raise TypeError("layers must be a list of positive integers")
        
        # Set private instance attributes
        self.__L = len(layers)  # Number of layers in the neural network
        self.__cache = {}  # Dictionary to hold all intermediary values
        self.__weights = {}  # Dictionary to hold all weights and biases
        
        # Initialize weights and biases for each layer
        prev_layer_size = nx  # Start with input size
        
        for layer_index in range(self.__L):
            current_layer_size = layers[layer_index]
            layer_num = layer_index + 1  # Layer numbering starts from 1
            
            # Initialize weights using He et al. method
            # He initialization: W = np.random.randn(layer_size, prev_layer_size) * sqrt(2 / prev_layer_size)
            self.__weights[f"W{layer_num}"] = np.random.randn(current_layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size)
            
            # Initialize biases to zeros
            self.__weights[f"b{layer_num}"] = np.zeros((current_layer_size, 1))
            
            # Update prev_layer_size for next iteration
            prev_layer_size = current_layer_size

    @property
    def L(self):
        """
        Getter method for the number of layers in the neural network

        Returns:
            int: The number of layers in the neural network
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter method for the cache dictionary

        Returns:
            dict: Dictionary to hold all intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter method for the weights dictionary

        Returns:
            dict: Dictionary to hold all weights and biases of the network
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples

        Returns:
            tuple: The output of the neural network and the cache
        """
        # Store input data in cache as A0
        self.__cache['A0'] = X
        
        # Initialize the current activation to the input data
        A_current = X
        
        # Forward propagation through all layers
        for layer_num in range(1, self.__L + 1):
            # Get weights and biases for current layer
            W = self.__weights[f'W{layer_num}']
            b = self.__weights[f'b{layer_num}']
            
            # Calculate linear transformation: Z = W * A_prev + b
            Z = np.matmul(W, A_current) + b
            
            # Apply sigmoid activation function: A = 1 / (1 + e^(-Z))
            A_current = 1 / (1 + np.exp(-Z))
            
            # Store activated output in cache
            self.__cache[f'A{layer_num}'] = A_current
        
        # Return the final output (A_L) and the cache
        return A_current, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            float: The cost of the model
        """
        # Number of examples
        m = Y.shape[1]
        
        # Calculate logistic regression cost function
        # Cost = -(1/m) * sum(Y*log(A) + (1-Y)*log(1-A))
        # Use 1.0000001 - A to avoid division by zero errors
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        
        return cost
