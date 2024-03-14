import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        self.activation = None
        self.z = None

    def forward_pass(self, inputs):
        self.z = np.dot(self.weights, inputs) + self.biases
        self.activation = sigmoid(self.z)
        return self.activation

    def backward_pass(self, delta, prev_activation, learning_rate):
        delta = delta * sigmoid_prime(self.z)
        nabla_b = delta
        nabla_w = np.dot(delta, prev_activation.transpose())
        prev_delta = np.dot(self.weights.transpose(), delta)
        self.weights -= learning_rate * nabla_w
        self.biases -= learning_rate * nabla_b
        return prev_delta

class NeuralNetwork:
    def __init__(self, shape):
        self.layers = []
        for i in range(1, len(shape)):
            self.layers.append(Layer(shape[i-1], shape[i]))

    def forward_pass(self, x):
        activation = x
        for layer in self.layers:
            activation = layer.forward_pass(activation)
        return activation

    def backward_pass(self, x, y, learning_rate):
        output = self.forward_pass(x)
        delta = self.cost_derivative(output, y)
        for i in range(len(self.layers) - 1, 0, -1):
            delta = self.layers[i].backward_pass(delta, self.layers[i-1].activation, learning_rate)
        self.layers[0].backward_pass(delta, x, learning_rate)

    def train(self, training_data, epochs, mini_batch_size, learning_rate):
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

    def update_mini_batch(self, mini_batch, learning_rate):
        for x, y in mini_batch:
            self.backward_pass(x, y, learning_rate)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
