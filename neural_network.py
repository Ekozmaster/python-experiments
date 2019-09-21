# Simple Dense Deep Neural Network.
# All layers are densely connected. You must initialize it with a list of neurons by layer:
# nn = NeuralNetwork([2, 4, 3])
# which means 2 input neurons, 3 output neurons, and 4 neurons on a hidden layer.
import numpy as np


# Each neuron on the network stores it's linear input (neurons_linear), it's activated value
# (eg.: neurons = ReLU(neurons_linear)), and the partial derivative of the cost function with respect to the neuron's
# activation (neurons_dCdA), which is used to propagate derivatives backwards between layers.
class NeuralNetwork:
    learningRate = 0.01
    training_data = []
    training_labels = []

    # Pass a list of neurons count on each layer (eg. [2,3,2] -> 2 input neurons, 3 in a hidden layer, 2 in output)
    def __init__(self, layers_setup):
        # Neurons
        layers_neurons = []
        for i in layers_setup:
            layers_neurons.append(np.zeros(i))
        self.neurons = np.array(layers_neurons)

        # Weights
        self.weights = []
        # He-et-al weights initialization:
        # random(neurons_in_prev + neurons_in_cur) * sqrt(2.0 / (neurons_in_prev + neurons_in_cur)).
        # Properly initializing weights in a network is a pretty important step. If not done appropriately it might
        # never converge to a solution, or take an eternity to learn, due to vanishing/exploding gradients and such.
        # A good quick read:
        # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        for i in range(1, len(layers_setup)):
            self.weights.append(np.random.randn(layers_setup[i], layers_setup[i - 1]) *
                                np.sqrt(2 / (layers_setup[i - 1] + layers_setup[i])))
        self.weights = np.array(self.weights)

        # Biases
        self.biases = np.array(self.neurons[1:])

        # Auxiliar Data Structures
        # weights/biases_feedbacks is used to average the gradient vector across the training samples.
        self.weights_feedback = np.array(self.weights * 0.0)
        self.biases_feedback = np.array(self.biases * 0.0)

        self.neurons_dCdA = np.array(self.neurons[1:] * 0)
        self.neurons_linear = np.array(self.neurons * 0)

    def feed_forward(self):
        for layer in range(1, len(self.weights) + 1):
            self.neurons_linear[layer] = np.matmul(self.weights[layer - 1], self.neurons[layer - 1]) + self.biases[layer - 1]
            self.neurons[layer] = self.leaky_relu(self.neurons_linear[layer])

    def train(self, batch_size=0):
        self.reset_aux_data()
        batch_size_normalizer = batch_size or len(self.training_data)  # If batch_size == 0, take whole data as a batch.
        # For every training sample
        for data_index in range(0, len(self.training_data)):
            # Feedforward
            self.neurons[0] = np.array(self.training_data[data_index])
            self.feed_forward()
            error = self.neurons[-1] - np.array(self.training_labels[data_index])

            # Derivative of cost function (error^2) with respect to each neuron in the last layer.
            self.neurons_dCdA[-1] = 2 * error

            # Backpropagation
            # For every layer backwards
            for layer in reversed(range(len(self.weights))):
                dAdZ_dCdA = np.array([self.d_leaky_relu(self.neurons_linear[layer + 1]) * self.neurons_dCdA[layer]])
                # Applying partial derivatives of the cost function with respect to biases (dCdB).
                self.biases_feedback[layer] -= dAdZ_dCdA.flatten()
                # With respect to weights (dCdW).
                self.weights_feedback[layer] -= np.matmul(dAdZ_dCdA.transpose(), np.array([self.neurons[layer]]))

                if layer > 0:
                    # Saving this dCdA to use in the next iteration to compute dCdWs and dCdBs for the previous layer.
                    self.neurons_dCdA[layer - 1] = np.matmul(self.weights[layer].transpose(), dAdZ_dCdA.transpose()).flatten()
            # If finished current batch, apply the weights/biases' feedback averaged across all the batch samples
            # to the NN attenuated by the Learning Rate.
            if (data_index + 1) % batch_size_normalizer == 0 or (data_index + 1) % len(self.training_data) == 0:
                self.weights += self.weights_feedback * self.learningRate / batch_size_normalizer
                self.biases += self.biases_feedback * self.learningRate / batch_size_normalizer
                self.reset_aux_data()

    @staticmethod
    def leaky_relu(activation):
        return np.maximum(activation, activation*0.01)

    @staticmethod
    def d_leaky_relu(activation):
        activation[activation <= 0] = 0.01
        activation[activation > 0] = 1.0
        return activation

    def reset_aux_data(self):
        self.weights_feedback *= 0
        self.biases_feedback *= 0

    # Set self.training_data/labels to the testing data/labels that you have before calling this method.
    def print_accuracy(self):
        correct = 0
        for data_index in range(0, len(self.training_data)):
            # Feedforward
            self.neurons[0] = np.array(self.training_data[data_index])
            self.feed_forward()

            if np.argmax(self.neurons[-1]) == np.argmax(self.training_labels[data_index]):
                correct += 1
        print("Accuracy: {0}/{1}: {2:.2f}%".format(correct, len(self.training_data), correct / len(self.training_data) * 100))

    def predict_for(self, sample):
        if len(sample) != len(self.neurons[0]):
            return -1
        self.neurons[0] = np.array(sample)
        self.feed_forward()
        return np.argmax(self.neurons[-1])
