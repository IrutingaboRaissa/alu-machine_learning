#!/usr/bin/env python3
"""
Module containing the Neuron class for binary classification
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

        # Initialize weights using random normal distribution
        # Shape is (1, nx) as shown in the example output
        self.W = np.random.normal(size=(1, nx))
        
        # Initialize bias to 0
        self.b = 0
        
        # Initialize activated output to 0
        self.A = 0
