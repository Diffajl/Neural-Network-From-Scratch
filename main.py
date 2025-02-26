import numpy as np

class NeuralNetwork:
    def __init__(self, n_neurons, n_inputs):
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs

    def init_params(self):
        self.weights = np.random.randn(self.n_inputs, self.n_neurons) 
        self.bias = np.random.randn(1, self.n_neurons)

    def activation(self, z):
        return 1 / (1 + np.exp(-z)) 
    
    def d_activation(self, z):
        return self.activation(z) * (1 - self.activation(z)) 

    def forward(self, X, y, lr, epoch):
        X = np.array(X)
        y = np.array(y)

        self.init_params()

        for i in range(epoch):
            z = np.dot(X, self.weights) + self.bias 
            self.y_pred = self.activation(z)

            error = np.mean((y - self.y_pred) ** 2)
            print(f"Epoch {i+1}/{epoch}, Predicted: {self.y_pred.T}, Error: {error}")

            d_loss = 2 * (self.y_pred - y) / len(X) 
            d_weights = np.dot(X.T, d_loss * self.d_activation(z)) 
            d_bias = np.sum(d_loss * self.d_activation(z), axis=0, keepdims=True)  

            self.weights -= lr * d_weights
            self.bias -= lr * d_bias

X = np.array([[0], [1], [2], [3]])  
y = np.array([[0], [1], [0], [1]]) 
nn = NeuralNetwork(n_neurons=1, n_inputs=1)  
nn.forward(X, y, lr=0.1, epoch=1000)  
