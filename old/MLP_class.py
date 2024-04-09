import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class MLP_regressor(object):
    """
    Class to encapsulate the MLP Regressor.
    """

    def __init__(self, lr: float=1e-2, epochs: int=100, layers: List[int]=[3,5,2]) -> None:
        """
        Constructor function for a MLP instance

        Inputs:
            lr: learning rate
            epochs: number of epochs to use during training
            layers: number of hidden layers of MLP
        """
        self.lr = lr
        self.epochs = epochs
        self.layers = layers
        self.weights = []
        self.biases = []
        self.loss = []

    def __del__(self) -> None:
        """
        Destructor function for a MLP instance
        """
        del self.lr
        del self.epochs
        del self.layers
        del self.weights
        del self.biases
        del self.loss

    def loss(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Function to compute the squared-error loss per sample

        Inputs:
            y_true: numpy array of true labels
            y_pred: numpy array of prediction value
        Outputs:
            loss value
        """
        return 0.5*(y_true - y_pred)**2
    
    def der_loss(self, y_true: np.array, y_pred: np.array) -> np.array:
        """
        Function to compute the derivative of the squared-error loss per sample

        Inputs:
            y_true: numpy array of true labels
            y_pred: numpy array of prediction values
        Outputs:
            derivative of loss value
        """
        return -(y_true - y_pred)
    
    # TODO: agrego relu?
    
    def sigmoid(self, z:np.array) -> np.array:
        """
        Function to compute sigmoid activation function

        Inputs: 
            z: input dot product (w*x + b)
        Outputs:
            determined sigmoid activation function
        """
        return 1/(1 + np.exp(-z))
    
    def der_sigmoid(self, z:np.array) -> np.array:
        """
        Function to compute the derivative of the sigmoid activation function

        Inputs:
            z: input dot product (w*x + b)
        Outputs:
            determined derivative of sigmoid activation function
        """
        return self.sigmoid(z)*(1 - self.sigmoid(z))
    
    def _linear(self, z: np.array) -> np.array:
        """
        Function to compute linear activation function
        
        Inputs:
            z: input dot product (w*x + b)
        Outputs:
            determined linear activation function
        """
        return z
    
    def der_linear(self, z:np.array) -> np.array:
        """
        Function to compute the derivative of the linear activation function
        
        Inputs:
            z: input dot product (w*x + b)
        Outputs:
            determined derivative of linear activation function
        """
        return np.ones(z.shape)
    
    def forward_pass(self, X: np.array) -> Tuple[List[np.array], List[np.array]]:
        """
        Function to perform forward pass through the MLP
        
        Inputs:
            X: numpy array of input predictive features with assumed shape [number_features, number_samples]
        Outputs:
            list of activations and derivatives for each layer
        """ 
        # Record Input Layer
        input_to_layer = np.copy(X)
        activations = [input_to_layer]
        derivatives = [np.zeros(X.shape)]

        # Hidden Layers
        for i in range(len(self.layers) - 2):
            z_i = np.dot(input_to_layer, self.weights[i]) + self.biases[i]
            input_to_layer = self.sigmoid(z_i)
            activations.append(input_to_layer)
            derivatives.append(self.der_sigmoid(z_i))

        # Output Layer
        z_i = np.dot(input_to_layer, self.weights[-1]) + self.biases[-1]
        input_to_layer = self._linear(z_i)
        activations.append(input_to_layer)
        derivatives.append(self.der_linear(z_i))

        return (activations, derivatives)
        
    def backward_pass(self, 
                       activations: List[np.array], 
                       derivatives: List[np.array], 
                       y: np.array) -> Tuple[List[np.array], List[np.array]]:
        """
        Function to perform backward pass through the network
        
        Inputs:
            activations: list of activations from each layer in the network
            derivatives: list of derivatives from each layer in the network
            y: numpy array of target values with assumed shape [output dimension, number_samples]
        Output:
            list of numpy arrays containing the derivatives of the loss function wrt layer weights
        """ 
        # Record loss
        self.loss.append((1/y.shape[1]) * np.sum(self.loss(y, activations[-1])))
        
        # Initialize lists to store derivatives of weights and biases
        d_weights = []
        d_biases = []
        
        # Compute derivatives for Output Layer
        dl_dy = self.der_loss(y, activations[-1])
        dl_dz = np.multiply(dl_dy, derivatives[-1])
        dl_dw = (1/y.shape[1]) * np.dot(dl_dz, activations[-2].T)
        dl_db = (1/y.shape[1]) * np.sum(dl_dz, axis=1)
        
        d_weights.append(dl_dw)
        d_biases.append(dl_db)
        
        # Compute derivatives for Hidden Layers
        for i in range(len(self.layers) - 2, 0, -1):
            dl_dy = np.dot(self.weights[i], dl_dz)
            dl_dz = np.multiply(dl_dy, derivatives[i])
            dl_dw = (1/y.shape[1]) * np.dot(dl_dz, activations[i-1].T)
            dl_db = (1/y.shape[1]) * np.sum(dl_dz, axis=1)
            
            d_weights.insert(0, dl_dw)
            d_biases.insert(0, dl_db)
        
        return (d_weights, d_biases)


    def update_weights(self, dl_dw: List[np.array], dl_db: List[np.array]) -> None:
        """
        Function to apply update rule to model weights and biases

        Inputs:
            dl_dw: list of numpy arrays containing loss derivatives wrt weights
            dl_db: list of numpy arrays containing loss derivatives wrt biases
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * dl_dw[i]
            self.biases[i] -= self.lr * dl_db[i].reshape(-1, 1)

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Function to train an MLP instance
        
        Inputs:
            X: numpy array of input predictive features with assumed shape [number_samples, number_features]
            y: numpy array of target values with assumed shape [number_samples, output dimension]
        """
        # Initialise the model parameters
        input_size = X.shape[1]
        self.weights.clear()
        self.biases.clear()
        self.loss.clear() 
        for idx in range(len(self.layers)-1):
            if idx == 0:
                self.weights.append(np.random.randn(self.layers[idx+1], input_size) * 0.1)
            else:
                self.weights.append(np.random.randn(self.layers[idx+1], self.layers[idx]) * 0.1)
            self.biases.append(np.random.randn(self.layers[idx+1], 1) * 0.1)      
        
        # Loop through each epoch
        for _ in range(self.epochs):
            # 1. Do forward pass through the MLP
            activations, derivatives = self.forward_pass(X.T)
            # 2. Do backward pass through the MLP
            dl_dw, dl_db = self.backward_pass(activations, derivatives, y.T)
            # 3. Update weights
            self.update_weights(dl_dw, dl_db)   

    def predict(self, X: np.array) -> np.array:
        """
        Function to produce predictions from a trained MLP instance
        
        Input:
            X: numpy array of input predictive features with assumed shape [number_samples, number_features]
        Output:
            numpy array of model predictions
        """
        # Do forward pass through the MLP
        activations, _ = self.forward_pass(X.T)
        # Return predictions
        return activations[-1].T


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
# generate data
X, y = make_regression(n_samples=10000, n_features=3, n_targets=2, noise=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MLP_regressor(lr=0.01, epochs=150, layers=[3,5,2])
model.fit(X_train, y_train)

plt.plot(model.loss)
plt.title("Training Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()