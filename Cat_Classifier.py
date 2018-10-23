#Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#Find values for m_train , m_test , num_px
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[2]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#standardize our dataset
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

#Building the parts of our algorithm
#Sigmoid function
def sigmoid(z):
	s = 1/(1+np.exp(-(z)))
	return s

#Initialize with zeros
def initialize_with_zeros(dim):
	w = np.zeros([dim,1])
	b = 0
	assert(w.shape == (dim, 1))
	assert(isinstance(b, float) or isinstance(b, int))
	return w, b


#Propogate Function
def propagate(w, b, X, Y):
	m = X.shape[1]

	#Forward Propogation
	A = sigmoid(np.dot(w.T,X) + b)
	cost = -1/m * (np.dot(Y,np.log(A).T) + np.dot((1-Y),np.log(1 - A).T))

	#Backward Propogation
	dw = 1 / m *(np.dot(X,(A - Y).T))
	db = 1 / m *(np.sum(A - Y))

	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())
	
	grads = {"dw": dw,
			 "db": db}
	
	return grads, cost


#Optimize Function
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
	costs = []
	
	for i in range(num_iterations):

		# Cost and gradient calculation
		grads, cost = propagate(w, b, X, Y)

		# Retrieve derivatives from grads
		dw = grads["dw"]
		db = grads["db"]

		# update rule
		w = w - learning_rate*dw
		b = b - learning_rate*db

		# Record the costs
		if i % 100 == 0:
			costs.append(cost)
		
		# Print the cost every 100 training iterations
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))
	
	params = {"w": w,
			  "b": b}
	
	grads = {"dw": dw,
			 "db": db}
	
	return params, grads, costs


#Predict Function
def predict(w, b, X):
	m = X.shape[1]
	Y_prediction = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)

	# Compute vector "A" predicting the probabilities of a cat being present in the picture
	A = sigmoid(np.dot(w.T,X) + b)

	for i in range(A.shape[1]):

		# Convert probabilities A[0,i] to actual predictions p[0,i]
		if(A[0][i] <= 0.5):
			Y_prediction[0][i] = 0
		else:
			Y_prediction[0][i] = 1

	assert(Y_prediction.shape == (1, m))
	
	return Y_prediction


#Model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.005, print_cost = False):

	# initialize parameters with zeros
	w, b = initialize_with_zeros(X_train.shape[0])

	# Gradient descent
	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations= 2000, learning_rate = 0.005, print_cost = False)

	# Retrieve parameters w and b from dictionary "parameters"
	w = parameters["w"]
	b = parameters["b"]

	# Predict test/train set examples
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)

	# Print train/test Errors
	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

	
	d = {"costs": costs,
		 "Y_prediction_test": Y_prediction_test, 
		 "Y_prediction_train" : Y_prediction_train, 
		 "w" : w, 
		 "b" : b,
		 "learning_rate" : learning_rate,
		 "num_iterations": num_iterations}
	
	return d


#Executing the Script
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
