#!/usr/bin/env python3
"""
Module containing the NeuralNetwork class for binary classification with private attributes
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

        # Initialize private weights for hidden layer using random normal distribution
        # __W1 has shape (nodes, nx) - each row is weights for one hidden node
        self.__W1 = np.random.normal(size=(nodes, nx))
        
        # Initialize private bias for hidden layer with zeros
        # __b1 has shape (nodes, 1) - one bias per hidden node
        self.__b1 = np.zeros((nodes, 1))
        
        # Initialize private activated output for hidden layer
        self.__A1 = 0
        
        # Initialize private weights for output neuron using random normal distribution
        # __W2 has shape (1, nodes) - weights from hidden layer to output
        self.__W2 = np.random.normal(size=(1, nodes))
        
        # Initialize private bias for output neuron
        self.__b2 = 0
        
        # Initialize private activated output for output neuron (prediction)
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter method for the weights vector for the hidden layer

        Returns:
            numpy.ndarray: The weights vector for the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter method for the bias for the hidden layer

        Returns:
            numpy.ndarray: The bias for the hidden layer
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter method for the activated output for the hidden layer

        Returns:
            int or numpy.ndarray: The activated output for the hidden layer
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter method for the weights vector for the output neuron

        Returns:
            numpy.ndarray: The weights vector for the output neuron
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter method for the bias for the output neuron

        Returns:
            int: The bias for the output neuron
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter method for the activated output for the output neuron

        Returns:
            int or numpy.ndarray: The activated output for the output neuron (prediction)
        """
        return self.__A2
