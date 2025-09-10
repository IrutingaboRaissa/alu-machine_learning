#!/usr/bin/env python3
"""
Module containing the Neuron class for binary classification with private attributes
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
            int: The activated output of the neuron (prediction)
        """
        return self.__A
