import random

from src.NNClassifier import *
from matplotlib import pyplot as plt

from src.DataWorker import *

path = "../data/house_prices/data.csv"
dataWorker = DataWorker(path)
data = dataWorker.get_data()

#create classifier
prepared_train_vector = data[0:1300]
prepared_test_vector = data[1300:]

nn_classifier = NNClassifier(prepared_train_vector, prepared_test_vector)

train_error_matrix, test_error_matrix = nn_classifier.train_network_get_error()


train_error_matrix = np.multiply(train_error_matrix, dataWorker.out_scale_order)
test_error_matrix = np.multiply(test_error_matrix, dataWorker.out_scale_order)

train_error_tuple = train_error_matrix
test_error_tuple = test_error_matrix
plt.figure()
x_axis_values = range(0, len(train_error_tuple))
# plt.plot(x_axis_values, train_error_tuple, 'r', x_axis_values, test_error_tuple, 'b')
train_line, = plt.plot(x_axis_values, train_error_tuple, 'r', label='train')
test_line, = plt.plot(x_axis_values, test_error_tuple, 'b', label='test')
plt.legend(handles=[train_line, test_line])
plt.title('smth smth error')
plt.ylabel('train error')
plt.xlabel('epoch')
plt.show()

