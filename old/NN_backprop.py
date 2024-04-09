import numpy as np

class MLP(object):

    def __init__(self, 
                 layers = [1,3,1],
                 activations = ["relu", "linear"],
                 lr = 1e-5, 
                 epochs = 10) -> None:
        """
        Constructor function for a MLP instance

        Inputs:
            layers: number of nodes in each layer of MLP
            activations: layers activation functions 
            lr: learning rate
            epochs: number of epochs to use during training
        """
        #self.loss = []
        assert len(layers) == len(activations) + 1, "Number of layers and activations mismatch" # Number of layers - 1 (input)= number of activation functions 
        self.layers = layers
        self.activations = activations
        self.epochs = epochs
        self.lr = lr

        """         
        The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers. 
        """
        #self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        #self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

        self.weights = []
        self.biases = []
        for i in range(len(layers)-1):
            self.weights.append(np.random.randn(layers[i+1], layers[i]))
            self.biases.append(np.random.randn(layers[i+1], 1))
    
    def forward_pass(self, x):
        """
        Returns the output of the MLP if x is the input.
        """
        print("x.shape:", x.shape)
        x = x.reshape(-1,1)
        z = [np.copy(x).transpose()]
        a = [] # list to store the preactivations, layer by layer
        print("Weights", self.weights,"\nZ:", z, "\nBiases", self.biases)
        for l in range(1, len(self.layers)):
            # Preactivation in layer l "a(l) = W(l)z(l-1) + b(l)"
            print(f"w shape:{self.weights[l-1].shape}, Z shape: {z[l-1].shape}")
            a_l = np.dot(self.weights[l-1], z[l-1]) + self.biases[l-1]
            print("shape al:", a_l.shape)
            a.append(a_l)

            # Activation in layer l (z(l) = h(a(l)))
            h = self.getActivationFunction(self.activations[l-1])
            z_l = h(a_l)
            z.append(np.copy(z_l))

        return a, z

    def backward_pass(self, z, a, y):
        # Initialize ẟ "error for each layer" dy|da
        delta = [np.zeros(w.shape) for w in self.weights]
        # Insert output layer error ẟ(L) is h'(a(L))** (z[-1] - y)
        delta[-1] = (y-z[-1])*self.getDerivitiveActivationFunction(self.activations[-1])(a[-1])

        # Initialize gradients dy/dW and dy/db
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Insert output layer values
        nabla_b[-1] = delta[-1]
        nabla_w[-1] = np.dot(delta[-1], z[-2].transpose())

        # Calculate deltas for hidden layers -> Backpropagation of errors
        for l in reversed(range(1, len(delta)-1)):
            # Calculate ẟ for layer l
            h_prime = self.getDerivitiveActivationFunction(self.activations[l])(a[l])
            delta[l] = np.dot(self.weights[l+1].transpose(), delta[l+1]) * h_prime

            # Calculate gradients
            nabla_b[l-1] = delta[l] 
            nabla_w[l-1] = np.dot(delta[l], z[l-1].transpose())
        return nabla_b, nabla_w

    def train(self, x, y, batch_size = 1):
        """
        Function to follow the sequence:
        for e in epochs:
            1. Forward pass
            2. Backward pass
            3. Update model weights
        """
        for e in range(self.epochs):
            # Mini-batch SGD 
            i=0
            while(i<len(y)):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                i = i+batch_size

                # 1. Forward pass
                z, a = self.forward_pass(x_batch)
                # 2. Backward pass
                nabla_b, nabla_w = self.backward_pass(z, a, y_batch)
                # 3. Update model weights
                self.weights = [w + self.lr * dweight for w, dweight in zip(self.weights, nabla_w)]
                self.biases = [w + self.lr * dbias for w, dbias in zip(self.biases, nabla_b)]
                print("loss = {}".format(np.linalg.norm(a[-1] - y_batch)))

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.forward_pass(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    @staticmethod # It belongs to the class, not its instances. It does not require an instance for it to be called.
    def getActivationFunction(name):
        if(name == 'sigmoid'):
            return lambda x : np.exp(x)/(1+np.exp(x))
        elif(name == 'linear'):
            return lambda x : x
        elif(name == 'relu'):
            return lambda x: np.maximum(x,0)
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
        else:
            print('Unknown activation function. Linear is used by default.')
            return lambda x: 1