import numpy as np
from matplotlib import pyplot as plt

class NNClassifierProxy:
    def __init__(self, nnClassifier, outScaleOrder):
        # create reference to classifier
        self.nnClassifier = nnClassifier
        self.outScaleOrder = outScaleOrder

    def train(self):
        # train_error, test_error = self.network.train()
        # saves it in fields
        self.train_error_matrix, self.test_error_matrix = self.nnClassifier.train_network_get_error()

    def paintAverageAccuracyGraph(self):
        # draws graphs of how error behaved with epochs
        self.train_error_matrix = np.multiply(self.train_error_matrix, self.outScaleOrder)
        self.test_error_matrix = np.multiply(self.test_error_matrix, self.outScaleOrder)

        plt.figure()
        x_axis_values = range(0, len(self.train_error_matrix))
        train_line, = plt.plot(x_axis_values, self.train_error_matrix, 'r', label='train')
        test_line, = plt.plot(x_axis_values, self.test_error_matrix, 'b', label='test')
        plt.legend(handles=[train_line, test_line])
        plt.title('smth smth error')
        plt.ylabel('train error')
        plt.xlabel('epoch')
        plt.show()



