#!/usr/bin/env python3
"""
Module containing the DeepNeuralNetwork class for multiclass classification with save/load functionality
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """
    A class that defines a deep neural network performing multiclass classification
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
            
            # Apply activation function
            if layer_num == self.__L:
                # For the output layer, use softmax activation for multiclass classification
                # Softmax: A_i = e^(Z_i) / sum(e^(Z_j)) for all j
                # Subtract max for numerical stability
                Z_stable = Z - np.max(Z, axis=0, keepdims=True)
                exp_Z = np.exp(Z_stable)
                A_current = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # For hidden layers, use sigmoid activation function: A = 1 / (1 + e^(-Z))
                A_current = 1 / (1 + np.exp(-Z))
            
            # Store activated output in cache
            self.__cache[f'A{layer_num}'] = A_current
        
        # Return the final output (A_L) and the cache
        return A_current, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using categorical cross-entropy

        Args:
            Y (numpy.ndarray): One-hot encoded correct labels with shape (classes, m)
            A (numpy.ndarray): Activated output with shape (classes, m)

        Returns:
            float: The cost of the model
        """
        # Number of examples
        m = Y.shape[1]
        
        # Calculate categorical cross-entropy cost function
        # Cost = -(1/m) * sum(sum(Y * log(A)))
        # Add small epsilon to prevent log(0)
        epsilon = 1e-7
        cost = -(1/m) * np.sum(Y * np.log(A + epsilon))
        
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples
            Y (numpy.ndarray): One-hot encoded correct labels with shape (classes, m)

        Returns:
            tuple: The neural network's prediction and the cost of the network
        """
        # Perform forward propagation to get predictions
        A, _ = self.forward_prop(X)
        
        # Convert predictions to one-hot format
        # Find the index of the maximum value for each example (axis=0)
        # Then create one-hot encoded predictions
        max_indices = np.argmax(A, axis=0)
        predictions = np.zeros_like(A)
        predictions[max_indices, np.arange(A.shape[1])] = 1
        
        # Calculate cost using the cost method
        cost = self.cost(Y, A)
        
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y (numpy.ndarray): One-hot encoded correct labels with shape (classes, m)
            cache (dict): Dictionary containing all intermediary values of the network
            alpha (float): Learning rate (default: 0.05)
        """
        # Number of examples
        m = Y.shape[1]
        
        # Start backpropagation from the output layer
        # For the last layer (output layer) with softmax: dZ_L = A_L - Y
        A_L = cache[f'A{self.__L}']
        dZ = A_L - Y
        
        # Backpropagate through all layers (from L to 1)
        for layer_num in range(self.__L, 0, -1):
            # Get the activation from the previous layer
            A_prev = cache[f'A{layer_num - 1}']
            
            # Calculate gradients for weights and biases
            # dW = (1/m) * dZ * A_prev.T
            dW = (1/m) * np.matmul(dZ, A_prev.T)
            
            # db = (1/m) * sum(dZ, axis=1, keepdims=True)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            
            # Update weights and biases using gradient descent
            # W = W - alpha * dW
            # b = b - alpha * db
            self.__weights[f'W{layer_num}'] = self.__weights[f'W{layer_num}'] - alpha * dW
            self.__weights[f'b{layer_num}'] = self.__weights[f'b{layer_num}'] - alpha * db
            
            # Calculate dZ for the previous layer (if not the first layer)
            if layer_num > 1:
                # Get weights from current layer
                W = self.__weights[f'W{layer_num}']
                
                # Get activation from previous layer
                A_prev = cache[f'A{layer_num - 1}']
                
                # Calculate dZ for previous layer using chain rule
                # dZ_prev = W.T * dZ * A_prev * (1 - A_prev)
                # The sigmoid derivative is A * (1 - A)
                dZ = np.matmul(W.T, dZ) * A_prev * (1 - A_prev)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the deep neural network

        Args:
            X (numpy.ndarray): Input data with shape (nx, m)
                nx is the number of input features
                m is the number of examples
            Y (numpy.ndarray): One-hot encoded correct labels with shape (classes, m)
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
            A_initial, _ = self.forward_prop(X)
            initial_cost = self.cost(Y, A_initial)
            print("Cost after 0 iterations: {}".format(initial_cost))
            
            # Add to graph data if graphing is enabled
            if graph:
                cost_history.append(initial_cost)
                iteration_history.append(0)
        
        # Training loop
        for i in range(iterations):
            # Forward propagation
            A, cache = self.forward_prop(X)
            
            # Gradient descent (backpropagation and parameter updates)
            self.gradient_descent(Y, cache, alpha)
            
            # Print cost at specified intervals
            if verbose and (i + 1) % step == 0:
                current_cost = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(i + 1, current_cost))
                
                # Add to graph data if graphing is enabled
                if graph:
                    cost_history.append(current_cost)
                    iteration_history.append(i + 1)
            elif graph and (i + 1) % step == 0:
                # If only graphing (not verbose), still collect cost data
                current_cost = self.cost(Y, A)
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

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Args:
            filename (str): The file to which the object should be saved
                           If filename does not have the extension .pkl, add it
        """
        # Add .pkl extension if not present
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        # Save the object using pickle
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Args:
            filename (str): The file from which the object should be loaded

        Returns:
            DeepNeuralNetwork or None: The loaded object, or None if filename doesn't exist
        """
        try:
            # Check if file exists
            if not os.path.exists(filename):
                return None
            
            # Load the object using pickle
            with open(filename, 'rb') as file:
                return pickle.load(file)
        
        except Exception:
            return None
