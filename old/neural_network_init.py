#!/usr/local/bin/python3

import numpy as np

from read_data import read_data

# def sigmoid(x):
# 	return 1.0 / (1 + np.exp(-x))


def sigmoid(signal):
    # Prevent overflow.
	# signal = np.clip( signal, -500, 500 )

    # Calculate activation signal
	signal = 1.0/(1 + np.exp(-signal))

	# print('sigmoid')

	return signal

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

def sigmoid_derivative(x):
	return x * (1.0 - x)

def mySum(x):
	s = 0
	for i in x:
		s += i

	return s


def softmax(x):
	exps = [np.exp(i) for i in x]
	sum_of_exps = mySum(exps)
	values = [j/sum_of_exps for j in exps]

	return np.asarray(values)


class NeuralNetwork:
	def __init__(self, x, y):
		self.input = x
		self.weights1 = np.random.rand(self.input.shape[1], 16)
		self.weights2 = np.random.rand(16, 10)
		self.y = y
		self.output = np.zeros(y.shape)

	def feedforward(self):
		# self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		# self.output = sigmoid(np.dot(self.layer1, self.weights2))
		# self.output = softmax(self.output)

		self.layer1 = ReLU(np.dot(self.input, self.weights1))
		self.output = ReLU(np.dot(self.layer1, self.weights2))

	def backprop(self):
		# d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output)
		                    # * sigmoid_derivative(self.output)))
		# d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) *
		                    # sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

		d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output)
		                    * dReLU(self.output)))
		d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) *
		                    dReLU(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

		self.weights1 += d_weights1
		self.weights2 += d_weights2

	# def backprop(self):
	# 	# backward propagate through the network
	# 	self.o_error = self.y - self.output # error in output
	# 	self.o_delta = self.o_error*sigmoid_derivative(self.output) # applying derivative of sigmoid to error

	# 	self.z2_error = self.o_delta.dot(self.weights2.T) # z2 error: how much our hidden layer weights contributed to output error
	# 	self.z2_delta = self.z2_error*sigmoid_derivative(self.layer1) # applying derivative of sigmoid to z2 error

	# 	self.weights1 += self.input.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
	# 	self.weights2 += self.layer1.T.dot(self.o_delta)

if __name__ == "__main__":
	x = np.array([[0,0,1],
				  [0,1,1],
				  [1,0,1],
				  [1,1,1]])
	y = np.array([[0],[1],[1],[0]])

	# np.seterr(all='ignore')

	# np.random.seed(1)

	x, y = read_data()

	print("read data")

	nn = NeuralNetwork(x, y)

	for i in range(20):
		print(i)
		nn.feedforward()
		nn.backprop()

	print(nn.output[0])

	# print(softmax([1.0, 2.0, 0.1]))

	print(mySum(nn.output[0]))
