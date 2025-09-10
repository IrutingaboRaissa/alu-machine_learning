#!/usr/bin/env python3
"""
Module containing the Neuron class for binary classification with evaluation
"""
import numpy as np


class Neuron:
    """
    A class that defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initialize a Neuron instance

        Args:
            nx (int): The number of input features to the neuron

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Initialize private weights using random normal distribution
        # Shape is (1, nx) as shown in the example output
        self.__W = np.random.normal(size=(1, nx))
        
        # Initialize private bias to 0
        self.__b = 0
        
        # Initialize private activated output to 0
        self.__A = 0

    @property
    def W(self):
        """
        Getter method for the weights vector

        Returns:
            numpy.ndarray: The weights vector for the neuron
        """
        return self.__W

    @property
    def b(self):
        """
        Getter method for the bias

        Returns:
            int: The bias for the neuron
        """
        return self.__b

    @property
    def A(self):
        """
        Getter method for the activated output

        Returns:
            numpy.ndarray: The activated output of the neuron (prediction)
        """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features to the neuron
                m is the number of examples

        Returns:
            numpy.ndarray: The activated output of the neuron (prediction)
        """
        # Calculate the linear combination: Z = W·X + b
        Z = np.dot(self.__W, X) + self.__b
        
        # Apply sigmoid activation function: A = 1 / (1 + e^(-Z))
        self.__A = 1 / (1 + np.exp(-Z))
        
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)

        Returns:
            float: The cost
        """
        # Get number of examples
        m = Y.shape[1]
        
        # Calculate logistic regression cost
        # Cost = -1/m * Σ[y*log(a) + (1-y)*log(1-a)]
        # Use 1.0000001 - A instead of 1 - A to avoid division by zero
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features to the neuron
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)

        Returns:
            tuple: (predictions, cost)
                predictions (numpy.ndarray): Binary predictions with shape (1, m)
                cost (float): The cost of the network
        """
        # Get the sigmoid outputs through forward propagation
        A = self.forward_prop(X)
        
        # Convert sigmoid outputs to binary predictions
        # 1 if A >= 0.5, 0 otherwise
        predictions = (A >= 0.5).astype(int)
        
        # Calculate the cost
        cost = self.cost(Y, A)
        
        return predictions, cost
