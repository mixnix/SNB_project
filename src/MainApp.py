import random

from matplotlib import pyplot as plt

from src.NNClassifier import *
from src.NNClassifierProxy import *
from src.DataWorker import *


path = "../data/house_prices/data.csv"
dataWorker = DataWorker(path)
data = dataWorker.get_data()

#create classifier
prepared_train_vector = data[0:1300]
prepared_test_vector = data[1300:]

nn_classifier = NNClassifier(prepared_train_vector, prepared_test_vector)

nn_proxy = NNClassifierProxy(nn_classifier, dataWorker.out_scale_order)

nn_proxy.train()

nn_proxy.paintAverageAccuracyGraph()


