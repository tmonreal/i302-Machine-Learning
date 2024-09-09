import numpy as np
import random

class MLP(object):

    def __init__(self, 
                 layers = [1,3,1],
                 activations = ["relu", "linear"]) -> None:
        """
        Constructor function for a MLP instance
        Inputs:
            layers: number of nodes in each layer of MLP
            activations: layers activation functions 
        """
        assert len(layers) == len(activations) + 1, "Number of layers and activations mismatch" 
        self.layers = layers
        self.num_layers = len(layers)
        self.activations = activations

        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    def forward_pass(self, x):
        """
        Returns the preactivations (a) and activations (z) 
        of each layer of the MLP.
        Inputs:
            x: input of MLP
        Returns: 
            a: preactivations of MLP layers 
            z: activations of MLP layers
        """
        z = [np.array(x).reshape(-1, 1)] # Activation of the first layer (x=z)
        a = [] # List to store the preactivations, layer by layer
        for l in range(1, self.num_layers):
            # Preactivation in layer l 
            a_l = np.dot(self.weights[l-1], z[l-1]) + self.biases[l-1]
            a.append(np.copy(a_l))

            # Activation in layer l 
            h = self.getActivationFunction(self.activations[l-1])
            z_l = h(a_l)
            z.append(np.copy(z_l))
        
        return a, z

    def backward_pass(self, a, z, y):
        """
        Backpropagation to calculate the gradients of the
        weights and biases.
        Inputs:
            a: preactivations of MLP layers 
            z: activations of MLP layers
            y: real output value
        Returns:
            nabla_b: gradients respect to biases
            nabla_b: gradients respect to weights
        """
        # Initialize ẟ "error for each layer" dy|da
        delta = [np.zeros(w.shape) for w in self.weights]
        h_prime = self.getDerivitiveActivationFunction(self.activations[-1])
        delta[-1] = (a[-1] - y)*h_prime(a[-1])

        # Initialize gradients with respect to weights and biases (▽W and ▽b)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Insert output layer values
        nabla_b[-1] = delta[-1]
        nabla_w[-1] = np.dot(delta[-1], z[-2].transpose())

        # Calculate deltas for hidden layers -> Backpropagation of errors
        for l in reversed(range(1, len(delta))):
            # Calculate delta for layer l
            h_prime = self.getDerivitiveActivationFunction(self.activations[l - 1])
            delta[l - 1] = np.dot(self.weights[l].transpose(), delta[l]) * h_prime(a[l - 1])
            # Calculate gradients
            nabla_b[l - 1] = delta[l - 1] 
            nabla_w[l - 1] = np.dot(delta[l - 1], z[l - 1].transpose())
        
        loss = np.sum((a[-1] - y) ** 2) / 2  # Mean squared error loss
        return nabla_b, nabla_w, loss

    def update_mini_batch(self, mini_batch, lr):
        """
        Updates the MLPs weights and biases by applying
        mini-batch gradient descent.
        Inputs:
            mini_batch: list of tuples (x,y)
            lr: learning rate
        Returns:
            average loss across mini-batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        total_loss = 0

        for x,y in mini_batch:
            # For each x,y we perform forward and back pass to get gradients
            # for the current sample
            delta_nabla_b, delta_nabla_w, loss = self.backward_pass(*self.forward_pass(x), y)
            # Update mini-batch gradients by summing the current sample gradient
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            total_loss += loss

        # Weights and bias updates for the MLP (substract lr*nabla mini batch)
        self.weights = [w - lr * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - lr*nb for b, nb in zip(self.biases, nabla_b)]
        return total_loss / len(mini_batch)

    def fit(self, training_data, epochs, mini_batch_size, lr, val_data=None):
        """
        Method to train MLP.
            Inputs are self explanatory
            Returns are self explanatory
        """
        train_losses = []
        val_losses = []
        n = len(training_data)
        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[i:i+mini_batch_size] for i in range(0,n,mini_batch_size)]

            for mini_batch in mini_batches:
                train_loss = self.update_mini_batch(mini_batch, lr)

            train_losses.append(train_loss)
            if val_data:
                val_loss = self.evaluate(val_data)
                val_losses.append(val_loss)
                print(f"Epoch {e}: Train Loss: {train_loss} | Val loss: {val_loss}")
            else:
                print(f"Epoch {e}: Train Loss: {train_loss}")
        
        return train_losses, val_losses
    

    def evaluate(self, test_data):
        sum_squared_error = 0
        for x, y in test_data:
            prediction = self.forward_pass(x)[-1][-1]  # Get the prediction from the last layer
            sum_squared_error += np.sum((prediction - y) ** 2)  # Compute squared error
        return sum_squared_error / len(test_data)  # Return mean squared error
    
    def predict(self, X):
        """
        Predicts the output values for input test data X 
        by performing forward propagation.
        Inputs:
            X: Input data (numpy.ndarray)
        Returns:
            Predicted output values (numpy.ndarray)
        """
        predictions = []
        for x in X:
            prediction = self.forward_pass(x)[-1][-1].flatten()
            predictions.append(prediction)
        return np.array(predictions)
    
    # It belongs to the class, not its instances. It does not require an instance for it to be called.
    @staticmethod 
    def getActivationFunction(name):
        if(name == 'sigmoid'):
            return lambda x : np.exp(x)/(1+np.exp(x))
        elif(name == 'linear'):
            return lambda x : x
        elif(name == 'relu'):
            return lambda x: np.maximum(x,0)
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        else:
            print('Unknown activation function. Linear is used by default.')
            return lambda x: x
        
    @staticmethod
    def getDerivitiveActivationFunction(name):
        if(name == 'sigmoid'):
            sig = lambda x : np.exp(x)/(1+np.exp(x))
            return lambda x :sig(x)*(1-sig(x)) 
        elif(name == 'linear'):
            return lambda x: 1
        elif(name == 'relu'):
            def relu_diff(x):
                y = np.copy(x)
                y[y>=0] = 1
                y[y<0] = 0
                return y
            return relu_diff
        elif name == 'tanh':
            return lambda x: 1 - np.tanh(x)**2
        else:
            print('Unknown activation function. Linear is used by default.')
            return lambda x: 1