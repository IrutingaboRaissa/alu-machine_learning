#!/usr/bin/env python3
"""
Module containing the NeuralNetwork class for binary classification
"""
import numpy as np


class NeuralNetwork:
    """
    A class that defines a neural network with one hidden layer performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Initialize a NeuralNetwork instance

        Args:
            nx (int): The number of input features
            nodes (int): The number of nodes found in the hidden layer

        Raises:
            TypeError: If nx or nodes is not an integer
            ValueError: If nx or nodes is less than 1
        """
        # Validate nx parameter
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Validate nodes parameter
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize weights for hidden layer using random normal distribution
        # W1 has shape (nodes, nx) - each row is weights for one hidden node
        self.W1 = np.random.normal(size=(nodes, nx))
        
        # Initialize bias for hidden layer with zeros
        # b1 has shape (nodes, 1) - one bias per hidden node
        self.b1 = np.zeros((nodes, 1))
        
        # Initialize activated output for hidden layer
        self.A1 = 0
        
        # Initialize weights for output neuron using random normal distribution
        # W2 has shape (1, nodes) - weights from hidden layer to output
        self.W2 = np.random.normal(size=(1, nodes))
        
        # Initialize bias for output neuron
        self.b2 = 0
        
        # Initialize activated output for output neuron (prediction)
        self.A2 = 0
