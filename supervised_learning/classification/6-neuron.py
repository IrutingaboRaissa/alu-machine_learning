#!/usr/bin/env python3
"""
Module containing the Neuron class for binary classification with training
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features to the neuron
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A (numpy.ndarray): Activated output with shape (1, m)
            alpha (float): Learning rate
        """
        # Get number of examples
        m = Y.shape[1]
        
        # Calculate gradients
        # dW = (1/m) * X * (A - Y)^T
        dW = (1/m) * np.dot(X, (A - Y).T)
        
        # db = (1/m) * sum(A - Y)
        db = (1/m) * np.sum(A - Y)
        
        # Update weights and bias
        self.__W = self.__W - alpha * dW.T
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features to the neuron
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)
            iterations (int): Number of iterations to train over
            alpha (float): Learning rate

        Raises:
            TypeError: If iterations is not an integer or alpha is not a float
            ValueError: If iterations is not positive or alpha is not positive

        Returns:
            tuple: (predictions, cost) - The evaluation of the training data
                   after iterations of training have occurred
        """
        # Validate iterations parameter
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        
        # Validate alpha parameter
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        # Training loop
        for i in range(iterations):
            # Forward propagation
            A = self.forward_prop(X)
            
            # Gradient descent
            self.gradient_descent(X, Y, A, alpha)
        
        # Return evaluation after training
        return self.evaluate(X, Y)
