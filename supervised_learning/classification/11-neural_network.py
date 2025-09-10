#!/usr/bin/env python3
"""
Module containing the NeuralNetwork class for binary classification with cost calculation
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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples

        Returns:
            tuple: The activated outputs (A1, A2) of the hidden layer and output layer
        """
        # Forward propagation for hidden layer
        # Z1 = W1 * X + b1
        Z1 = np.matmul(self.__W1, X) + self.__b1
        
        # Apply sigmoid activation function to get A1
        # sigmoid(z) = 1 / (1 + e^(-z))
        self.__A1 = 1 / (1 + np.exp(-Z1))
        
        # Forward propagation for output layer
        # Z2 = W2 * A1 + b2
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        
        # Apply sigmoid activation function to get A2 (final output)
        self.__A2 = 1 / (1 + np.exp(-Z2))
        
        # Return both activated outputs
        return self.__A1, self.__A2

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
