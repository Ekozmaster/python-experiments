import matplotlib.pyplot as plt
import numpy as np

# Data
data = np.array([(0.5, 3.0), (1.0, 2.0), (1.0, 1.0), (1.5, 1.5), (2.0, 1.0), (3.0, 1.0),
                (1.0, 4.0), (2.0, 3.0), (2.0, 4.0), (2.5, 3.0), (3.0, 1.5), (4.0, 1.5), (4.5, 1.0)])
labels = np.array([(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),
                  (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)])


def plot_data():
    for index, point in enumerate(data):
        shape = '^'
        color = 'g'
        if labels[index][0] == 0:
            shape = 'o'
            color = 'r'
        plt.plot(point[0], point[1], shape, color=color)
    plt.xlim(0, np.max(data[:, 0]) + 1)
    plt.ylim(0, np.max(data[:, 1]) + 1)
    plt.show()


# Network
inputNeurons = np.array([[0], [0]])
outputNeurons = np.array([[0], [0]])

weights = np.mat([np.random.normal(0.0, 1.0, 2), np.random.normal(0.0, 1.0, 2)])
biases = np.array([np.random.normal(0.0, 1.0, 2), np.random.normal(0.0, 1.0, 2)])


def relu(activation):
    return np.maximum(activation, 0)


def prediction(input):
    return relu(weights * input)


def feed_network(input_data):
    return np.mat(prediction(input_data)[0]).transpose()
    # label = np.mat(labels[0]).transpose()
    # print("Input: \n" + str(inputData))
    # print("Output: \n" + str(outputNeurons))
    # print("Label: \n" + str(label))
    # error_rms = np.power(label - outputNeurons, 2)
    # return error_rms


def run():
    training_index = 0
    input_data = np.mat(data[training_index]).transpose()
    output = feed_network(input_data)
    label = np.mat(labels[training_index]).transpose()
    error_rms = np.power(label - output, 2)

    grad = -2*(label - output)*input_data.transpose()
    print(output)
    #step = np.divide(error_rms, grad)
    print(np.sum(error_rms))

    #print(del_c_del_w)

#print(weights)
#run()
plot_data()
