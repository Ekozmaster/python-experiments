data = [(0.5, 3.0), (1.0, 2.0), (1.0, 1.0), (1.5, 1.5), (2.0, 1.0), (3.0, 1.0),
        (1.0, 4.0), (2.0, 3.0), (2.0, 4.0), (2.5, 3.0), (3.0, 1.5), (4.0, 1.5), (4.5, 1.0)]
labels = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),
          (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]


class NN:
    inputs = [0, 0]
    outputs = [0, 0]
    weights = [1.0, 1.0, 1.0, 1.0]
    bias = [0, 0]

    def predict(self):
        self.outputs[0] = self.relu(self.inputs[0] * self.weights[0] + self.bias[0]) + self.relu(self.inputs[1] * self.weights[1] + self.bias[0])
        self.outputs[1] = self.relu(self.inputs[0] * self.weights[2] + self.bias[1]) + self.relu(self.inputs[1] * self.weights[3] + self.bias[1])

    @staticmethod
    def relu(activation):
        return max(activation, 0)

    @staticmethod
    def d_relu(activation):
        return 1.0 if activation > 0 else 0.0

    def train(self):
        for epoch in range(10000):
            weights_feedback = [0, 0, 0, 0]
            biases_feedback = [0, 0]
            cost = [0, 0]
            for training_index in range(len(data)):
                self.inputs = list(data[training_index])
                self.predict()
                error = [self.outputs[0] - labels[training_index][0], self.outputs[1] - labels[training_index][1]]
                for i in range(len(error)):
                    cost[i] += error[i]

                weights_feedback[0] -= (2.0 * error[0] * self.inputs[0] * self.d_relu(self.outputs[0]))
                weights_feedback[1] -= (2.0 * error[0] * self.inputs[1] * self.d_relu(self.outputs[0]))
                weights_feedback[2] -= (2.0 * error[1] * self.inputs[0] * self.d_relu(self.outputs[1]))
                weights_feedback[3] -= (2.0 * error[1] * self.inputs[1] * self.d_relu(self.outputs[1]))
                biases_feedback[0] -= (2.0 * error[0]) * self.d_relu(self.outputs[0])
                biases_feedback[1] -= (2.0 * error[1]) * self.d_relu(self.outputs[1])

            for i in range(len(self.weights)):
                #print(str(self.weights[i]) + ' ' + str((weights_feedback[i] / float(len(data))) * 0.01))
                self.weights[i] += (weights_feedback[i] / float(len(data))) * 0.01
            for i in range(len(self.bias)):
                self.bias[i] += (biases_feedback[i] / float(len(data))) * 0.01

            #print('#')
            cost = [a / len(data) for a in cost]
            #print(cost)
            #print(str(self.weights))


n = NN()

# TEST BEFORE
n.inputs = list(data[0])
n.predict()
print(n.outputs)

n.train()

# TEST AFTER
n.inputs = list(data[0])
n.predict()
print(n.outputs)

print(n.weights)
print(n.bias)
#import nn