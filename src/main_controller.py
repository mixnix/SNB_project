from src.helperfunctions import *
from src.neural_network_classifier import *
from matplotlib import pyplot as plt

data = load_preprocess_house_data("../data/house_prices/data.csv")

scale_features = True
scale_target = True

epochs = 100

scale_factor = 0

####################################

train_data_and_results = data[0:1300]
test_data_and_results = data[1300:]

train_data = train_data_and_results[:, :-1]
train_results = train_data_and_results[:, -1:]
test_data = test_data_and_results[:, :-1]
test_results = test_data_and_results[:, -1:]

if scale_features:
    train_data = standarize(train_data)
    test_data = standarize(test_data)

if scale_target:
    train_results, test_results, scale_factor = scale_target_function(train_results, test_results)



prepared_train_vector = np.append(train_data, train_results, axis=1)
prepared_test_vector = np.append(test_data, test_results, axis=1)






# nn_classifier = NeuralNetworkClassifier(prepared_train_vector, prepared_test_vector, hidden_neurons=[15, 30, 50],
#                                         learning_rates=[0.03, 0.1, 0.3], epochs=epochs,
#                                         number_of_inputs=2, number_of_categories=2)
# train_accuracy, test_accuracy = nn_classifier.train_show_accuracy()
# for i in range(0, len(train_accuracy)):
#     train_accuracy_tuple = train_accuracy[i]
#     test_accuracy_tuple = test_accuracy[i]
#     plt.figure()
#     x_axis_values = range(0, len(train_accuracy_tuple[0]))
#     plt.plot(x_axis_values, train_accuracy_tuple[0], 'r', x_axis_values, test_accuracy_tuple[0], 'b')
#     plt.title(train_accuracy_tuple[1])
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.show()


#create classifier
nn_classifier = NeuralNetworkClassifier(prepared_train_vector, prepared_test_vector, hidden_neurons=[5],
                                        learning_rates=[0.03], epochs=epochs,
                                        number_of_inputs=10, number_of_outputs=1)

train_error_matrix, test_error_matrix = nn_classifier.train_show_error()

#sprawdz accuracy na poczatku

#plot error


for i in range(0, len(train_error_matrix)):
    train_error_tuple = train_error_matrix[i]
    test_error_tuple = test_error_matrix[i]
    plt.figure()
    x_axis_values = range(0,len(train_error_tuple[0]))
    plt.plot(x_axis_values, train_error_tuple[0], 'r', x_axis_values, test_error_tuple[0], 'b')
    plt.title(train_error_tuple[1])
    plt.ylabel('train error')
    plt.xlabel('epoch')
    plt.show()

#test accuracy
# accuracy_table = nn_classifier.all_networks_train_accuracy()
# for accuracy_tuple in accuracy_table:
#     print("train accuracy: " + str(accuracy_tuple[0]) + " " + accuracy_tuple[1])
#
# accuracy_table = nn_classifier.all_networks_test_accuracy()
# for accuracy_tuple in accuracy_table:
#     print("test accuracy: " + str(accuracy_tuple[0]) + " " + accuracy_tuple[1])
