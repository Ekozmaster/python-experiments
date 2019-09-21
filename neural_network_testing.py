from neural_network import NeuralNetwork
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# Use this to visualize a 2D NN (2 input neurons) with realtime plot of it's decision boundaries.
def test_2d_nn():
    # 2D data used to visualize network's decision boundary progressing.
    data = [(0.5, 3.0), (1.0, 2.0), (1.0, 1.0), (1.5, 1.5), (2.0, 1.0), (-0.5, 1.5), (0.5, 3.0), (1.0, 2.0), (1.0, 1.0), (1.5, 1.5), (2.0, 1.0), (-0.5, 1.8),  # red dots
            (1.0, 4.0), (2.0, 3.0), (2.0, 4.0), (2.5, 3.0), (3.0, 1.5), (4.0, 1.5), (4.5, 1.0), (-0.5, -0.5), (0.0, -0.5), (0.5, -0.25), (-1.5, 1.5), (-1.8, 4.0)]  # green dots
    labels = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),  # red dots
              (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]  # green dots

    def plot_nn(nn):
        image_size = (100, 100)
        img_arr = np.array(Image.new('RGB', image_size, color='black')).astype(np.float32)

        for i in range(0, image_size[0]):
            for j in range(0, image_size[1]):
                # pixels' range mapped from -5 to 5
                pos_x = float(i - image_size[0] / 2.0) / (image_size[0] / 2.0) * 5.0
                pos_y = float(j - image_size[1] / 2.0) / (image_size[1] / 2.0) * 5.0
                prediction = nn.predict_for([pos_x, pos_y])
                if prediction == 0:
                    img_arr[j, i] = [1.0, 0.5, 0.5]
                else:
                    img_arr[j, i] = [0.5, 1.0, 0.5]
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)

        img = Image.fromarray((img_arr[::-1] * 255).astype('uint8'))
        plt.imshow(img, extent=[-5, 5, -5, 5])

        for i in range(len(data)):
            color = 'r'
            if labels[i] == (0, 1):
                color = 'g'
            plt.plot([data[i][0]], [data[i][1]], color=color, marker='o')

    # Used to capture mouse events to add dots in the dataset (LMB = red, RMB = green).
    def event_handler(event):
        if not event.xdata or not event.ydata:
            return
        if event.button == 1:
            data.append((event.xdata, event.ydata))
            labels.append((1, 0))
        elif event.button == 3:
            data.append((event.xdata, event.ydata))
            labels.append((0, 1))

    import time
    neural_net = NeuralNetwork([2, 6, 5, 2])
    neural_net.training_data = data
    neural_net.training_labels = labels
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('button_press_event', event_handler)

    for frame in range(100):
        plt.cla()
        time0 = time.time()
        for epoch in range(100):
            neural_net.train()
        print("Time: {0}".format(time.time() - time0))
        plot_nn(neural_net)
        plt.pause(0.05)


# Test a network with MNIST's handwritten digits database.
def test_mnist_nn():
    # Loading mnist database.
    # You need to download this .npz file cause i'm not including it on git.
    # https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    mnist = np.load('local-resources/mnist.npz')

    # Training data
    x_train = np.array(mnist['x_train'])
    x_train = x_train.reshape((len(x_train), 784)) / 255
    y_train_indexes = np.array(mnist['y_train'])
    y_train = np.array([[0]*10]*len(y_train_indexes))
    for i in range(len(y_train)):
        y_train[i][y_train_indexes[i]] = 1.0

    # Testing data
    x_test = np.array(mnist['x_test'])
    x_test = x_test.reshape((len(x_test), 784)) / 255
    y_test_indexes = np.array(mnist['y_test'])
    y_test = np.array([[0]*10]*len(y_test_indexes))
    for i in range(len(y_test)):
        y_test[i][y_test_indexes[i]] = 1.0

    def train_nn(nn, epochs=1, batch_size=200):
        print("Training: {0} epochs, batch size = {1} samples.".format(epochs, batch_size))
        nn.training_data = x_train
        nn.training_labels = y_train
        print("Training...")
        for epoch in range(epochs):
            nn.train(batch_size=batch_size)
            print("Completed epoch {0} of {1}.".format(epoch + 1, epochs))

    def test_nn(nn):
        print("Testing Accuracy with {0} samples...".format(len(x_test)))
        nn.training_data = x_test
        nn.training_labels = y_test
        nn.print_accuracy()

    def load_nn(w_file='', b_file=''):
        nn = NeuralNetwork([784, 16, 16, 10])
        if w_file and b_file:
            weights = np.load(w_file, allow_pickle=True)
            biases = np.load(b_file, allow_pickle=True)
            nn.weights = weights
            nn.biases = biases
        return nn

    # Use this line to load a NN once you saved it in .npy files.
    # neural_net = load_nn('local-resources/nn_weights3.npy', 'local-resources/nn_biases3.npy')

    # Otherwise load a new nn:
    neural_net = load_nn()

    test_nn(neural_net)
    train_nn(neural_net, epochs=2, batch_size=1500)
    test_nn(neural_net)
    # np.save('local-resources/nn_weights3', neural_net.weights)
    # np.save('local-resources/nn_biases3', neural_net.biases)
