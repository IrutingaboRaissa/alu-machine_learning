#!/usr/bin/env python3
"""
Module containing the NeuralNetwork class for binary classification with enhanced training
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)

        Returns:
            tuple: The neural network's prediction and the cost of the network
        """
        # Perform forward propagation to get predictions
        _, A2 = self.forward_prop(X)
        
        # Convert predictions to binary labels (0 or 1)
        # If output >= 0.5, predict 1; otherwise predict 0
        predictions = np.where(A2 >= 0.5, 1, 0)
        
        # Calculate cost using the cost method
        cost = self.cost(Y, A2)
        
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)
            A1 (numpy.ndarray): Output of the hidden layer with shape (nodes, m)
            A2 (numpy.ndarray): Predicted output with shape (1, m)
            alpha (float): Learning rate (default: 0.05)
        """
        # Number of examples
        m = Y.shape[1]
        
        # Gradient for output layer (layer 2)
        # dZ2 = A2 - Y (derivative of cost with respect to Z2)
        dZ2 = A2 - Y
        
        # Gradient for W2: dW2 = (1/m) * dZ2 * A1.T
        dW2 = (1/m) * np.matmul(dZ2, A1.T)
        
        # Gradient for b2: db2 = (1/m) * sum(dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Gradient for hidden layer (layer 1)
        # dZ1 = W2.T * dZ2 * A1 * (1 - A1) (chain rule with sigmoid derivative)
        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        
        # Gradient for W1: dW1 = (1/m) * dZ1 * X.T
        dW1 = (1/m) * np.matmul(dZ1, X.T)
        
        # Gradient for b1: db1 = (1/m) * sum(dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        # Update weights and biases using gradient descent
        # W = W - alpha * dW
        # b = b - alpha * db
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples
            Y (numpy.ndarray): Correct labels with shape (1, m)
            iterations (int): Number of iterations to train over (default: 5000)
            alpha (float): Learning rate (default: 0.05)
            verbose (bool): Whether to print training information (default: True)
            graph (bool): Whether to plot training cost (default: True)
            step (int): Step size for verbose/graph output (default: 100)

        Returns:
            tuple: The evaluation of the training data after training

        Raises:
            TypeError: If iterations is not an integer, alpha is not a float, or step is not an integer
            ValueError: If iterations or alpha is not positive, or step is not valid
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
        
        # Validate step parameter only if verbose or graph is True
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        
        # Initialize lists for tracking cost if graphing is enabled
        if graph:
            cost_history = []
            iteration_history = []
        
        # Print initial cost if verbose is enabled
        if verbose:
            # Calculate initial cost (0th iteration)
            _, A2_initial = self.forward_prop(X)
            initial_cost = self.cost(Y, A2_initial)
            print("Cost after 0 iterations: {}".format(initial_cost))
            
            # Add to graph data if graphing is enabled
            if graph:
                cost_history.append(initial_cost)
                iteration_history.append(0)
        
        # Training loop
        for i in range(iterations):
            # Forward propagation
            A1, A2 = self.forward_prop(X)
            
            # Gradient descent (backpropagation and parameter updates)
            self.gradient_descent(X, Y, A1, A2, alpha)
            
            # Print cost at specified intervals
            if verbose and (i + 1) % step == 0:
                current_cost = self.cost(Y, A2)
                print("Cost after {} iterations: {}".format(i + 1, current_cost))
                
                # Add to graph data if graphing is enabled
                if graph:
                    cost_history.append(current_cost)
                    iteration_history.append(i + 1)
            elif graph and (i + 1) % step == 0:
                # If only graphing (not verbose), still collect cost data
                current_cost = self.cost(Y, A2)
                cost_history.append(current_cost)
                iteration_history.append(i + 1)
        
        # Plot training cost if graph is enabled
        if graph:
            plt.figure()
            plt.plot(iteration_history, cost_history, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        
        # Return evaluation of training data after all iterations
        return self.evaluate(X, Y)
