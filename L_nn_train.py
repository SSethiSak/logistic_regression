
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import model
from lr_utils import load_dataset
import pickle
#import model2
import L_layer_nn




train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = L_layer_nn.load_data()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

layer_dim = [12288, 7, 4, 1]
Neural_network_model = L_layer_nn.L_layer_model(train_set_x, train_set_y, layer_dim,num_iterations=4000,learning_rate= 0.013, print_cost=True)
file_name = 'my_neural_network_model.sav'
pickle.dump(Neural_network_model, open(file_name, 'wb'))
