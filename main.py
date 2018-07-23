import cPickle
import gzip
import numpy as np
import time

# Given some vector (y_1, ..., y_m) of labels indicating some class number (in our case then digits), build OHE-matrix
# with rows as samples, i.e. y_{ij} = 1 iff y_i = j
def oneHotEncode(labels):
	maxClass = np.max(labels)
	return np.matrix([[1 if label == i else 0 for i in range(maxClass+1)] for label in labels])

# normalizes each feature of the training data by setting mean to 0 and variance to 1.
# applies the same changes to test
def normalize(train, test):
	# get the column mean
	columnMean = train.mean(axis = 0)
	stdDeviation = train.std(axis = 0)
	for index, x in np.ndenumerate(stdDeviation):

		# avoid division by zero later
		if x < 1e-15:
			stdDeviation[index] = 1.

	train -= columnMean
	test -= columnMean
	train /= stdDeviation
	test /= stdDeviation

	return (train,test)

# This is the network class, which keeps track of the layers and handles feed-forward, backward propagation
# and the loss function. It can only handle cross-entropy right now.
class ClassificationNeuralNetwork:

	layers = [] 					# Here we store all layers
	numOfSamples = None 			# Full number of samples
	numOfFeatures = None 			# Full number of features
	numOfClasses = None				# Number of classes
	learningRate = None				# The learning rate to use for SGD
	X = None 						# full data set used for training
	Y = None 						# OHE-matrix for full training data set
	decayLR = True 					# if true, the learning rate will decay over time linearly.
	numOfEpochs = 0					# number of learning iterations
	crossEntropyDerivative = None # keeps the (1,m,K)-tensor storing the derivative of cross-entropy function
	
	# learningRate
	# X : numpy matrix of data with rows as samples, normalized to mean 0 and std dev. 1
	# Y : OHE encoded matrix for classes
	# batchSize: Batch size for learning
	def __init__(self, learningRate, X, Y, batchSizeForSGD, learnEpochs, decayLR = True):
		self.numOfSamples, self.numOfFeatures = np.shape(X)
		self.learningRate = learningRate
		self.X = X
		self.Y = Y
		self.numOfSamples, self.numOfClasses = np.shape(Y)
		self.batchSizeForSGD = batchSizeForSGD
		self.decayLR = decayLR
		self.numOfEpochs = learnEpochs
		self.crossEntropyDerivative = np.zeros((1,batchSizeForSGD,self.numOfClasses))

	# adds a hidden layer with the given dimension to the network
	def addHiddenLayer(self, dimension, activationFunction = "relu"):
		preDimension = self.layers[-1].dimension if len(self.layers) > 0 else self.numOfFeatures
		self.layers.append(NetworkLayer(preDimension, dimension, self.batchSizeForSGD, activationFunction))

	# the output layer, activation function is of course softmax
	def addOutputLayer(self):
		self.addHiddenLayer(self.numOfClasses, "softmax")

	def learn(self):
		print("Start learning...")

		for epoch in range(self.numOfEpochs):
			print("Epoch: " + str(epoch + 1) + "/" + str(self.numOfEpochs))
			
			# calculate factor to multiply the learning rate with (for this iteration)
			learningRateFactor = 1. if not self.decayLR else 1. - ((1. * epoch)/(1. * self.numOfEpochs)) # slowly decay learning rate
			self.learnEpoch(learningRateFactor)


	# One full learning iteration
	def learnEpoch(self, learningRateFactor):
		startTime = time.time() # measure how long we need

		# choose some random samples
		batchChoice = np.random.choice(self.numOfSamples, self.batchSizeForSGD, replace = False)
		input_batch = self.X[batchChoice,:]
		Y_batch = self.Y[batchChoice,:]

		# Feed forward: Pipe the output of one layer to the next
		for layer in self.layers:
			input_batch = layer.feedForward(input_batch)

		# calculate error in input batch
		batchError = self.cross_entropy(Y_batch, input_batch)

		print("Average Batch Error: " + str(batchError / (self.batchSizeForSGD * 1.)))

		# do back propagation. The first derivative we pass to the last layer (which will be the output layer)
		# is the derivative of the loss function. As the loss function takes a (m,K)-tensor (K = num of classes, m =
		# num of samples in batch), this will be a (1, m, K)-tensor
		lastDerivative = self.cross_entropy_derivative(Y_batch, input_batch);

		for i, layer in enumerate(reversed(self.layers)):
			# every layer outputs a (1, m, h_{i-1})-tensor, where h_i is the dimension of the i-th layer
			lastDerivative = layer.backPropagate(lastDerivative, self.learningRate * learningRateFactor, i is len(self.layers) - 1)

		print("Iteration took: " + str(time.time() - startTime))

		

	# given some arbitrary data, only feed forward and get predictions
	# as well as error (however only predictions are currently returned)
	def infer(self, dataX, dataY):
		# feed forward
		for layer in self.layers:
			dataX = layer.feedForward(dataX, storeSteps = False)

		# calculate error in input batch
		batchError = self.cross_entropy(dataY, dataX)
		return dataX

	# simple cross entropy implementation
	def cross_entropy(self, targetY, predictedX):
		return -1. * np.sum(np.multiply(targetY, np.log(predictedX + 1e-40)))

	# returns a (1,m,K) tensor, where m is number of samples and K number of classes
	def cross_entropy_derivative(self, targetY, predictedX):
		self.crossEntropyDerivative[0] = -1. * np.multiply(targetY, 1/(predictedX + 1e-15)) # add small margin to avoid 0-division
		return self.crossEntropyDerivative


# this is the class of a network layer, which handles in itself feed-forward and backpropagation
# m 	= number of samples in a batch
# h_i 	= the dimension of the i-th layer
class NetworkLayer:

	# For forward and backword propagation, we need to store multiple tensors
	WeightMat = None 				# the weight matrix, a (h_{i-1}, h_i)-matrix/tensor
	Bias = None						# The bias for this layer, (h_i)-tensor
	
	Forward_Input = None			# the feed forward input obtained from the previous layer, this is a (m, h_{i-1})-tensor
	Forward_MatProdPlusBias = None 	# this is the matrix product ForwardInput*WeightMat given some input I
	Forward_Activation = None 		# the activation function applied to the matrix product. This will also be the ouput that
									# will get piped further through the network
	Backprop_Input = None			# the backpropagation input obtained from the next layer (derivative of loss function
									# w.r.t h_{i+1}-layer). This is a (1, m, h_i)-tensor
	Backprop_Derivative = None	# the derivative of loss function w.r.t weight matrix which is used for SGD update

	ActivationDerivative = None		# Derivative of activation function. 
	BiasDerivative = None			# Derivative of bias addition with respect to bias. This is a (m, h_i, h_i)-tensor,
									# which is static.

	# During backpropagation we will need to take two derivatives of the matrix multiplication Input*WeightMatrix:
	# Once with respect to the Input (this will be passed on to the previous layer) and once w.r.t to the WeightMatrix
	# (this will be used for gradient updating). This is reflected in these two derivatives, where A and B refer to
	# the part in the multiplication A*B.
	MatADerivative = None			
	MatBDerivative = None

	def __init__(self, preDimension, dimension, batchSizeForSGD, activationFunction):
		self.h = dimension
		self.h_pre = preDimension
		self.dimension = dimension
		self.activationFunction = activationFunction
		self.batchSizeForSGD = batchSizeForSGD
		self.m = batchSizeForSGD

		# build up the weight matrix for this layer
		self.initializeWeightMatrixAndBias()

	def initializeWeightMatrixAndBias(self):
		self.WeightMat = np.matrix([[np.random.uniform()/50. for i in range(self.h)] for j in range(self.h_pre)])
		self.Bias = np.array([0.001 for i in range(self.h)]) # set the bias to some small constant

		# Let A be a (m, h_i)-tensor, which we add the bias to
		# the derivative of A+Bias with respect to Bias will always be the same, so we can already
		# build it up here.
		self.BiasDerivative = np.zeros((self.m, self.h, self.h))
		for a in range(self.m):
			for b in range(self.h):
				self.BiasDerivative[a,b,b] = 1.

	# passing data X (this is a (m, h_{i-1})-tensor with m being the number of samples)
	# through the layer
	# 'storeSteps' indicates if we should keep track of all the intermediate steps and store the tensors
	# which we will not need if we just want to compute the loss for the whole data, for example.
	# For backpropagation this is needed though
	def feedForward(self, X, storeSteps = True):

		# store the input
		self.Forward_Input = X

		# do the matrix multiplication and the bias row-wise (numpy does that automatically)
		self.Forward_MatProdPlusBias = self.Forward_Input*self.WeightMat + self.Bias

		# apply the activation function
		if self.activationFunction is "relu":
			out = self.relu(self.Forward_MatProdPlusBias)
		elif self.activationFunction is "softmax":
			out = self.softmax(self.Forward_MatProdPlusBias)
		else:
			raise ValueError("Unknown activation function in feed forward")

		if not storeSteps:
			self.Forward_MatProdPlusBias = None
			self.Forward_Input = None

		# return output so that it can be piped to next layer.
		return out

	# We start with the last derivative of loss function w.r.t. output from layer above,
	#this is a (1, m, h_i)-tensor
	def backPropagate(self, lastDerivative, learningRate, skipOutput = False):
		
		start = time.time()

		# using what we put out in feed forward, calculate derivative of activation function
		# and get DelLG
		if self.activationFunction is "relu":
			self.relu_derivative(self.Forward_MatProdPlusBias)
		elif self.activationFunction is "softmax":
			self.softmax_derivative(self.Forward_MatProdPlusBias)
		else:
			raise ValueError("Unknown activation function in Backprop")
	
		# Now begin the tensordot-products, which I think slows down the process a lot.

		# output: (1, m, h_i)
		DelLG = np.tensordot(lastDerivative, self.ActivationDerivative, axes = 2) 

		# output: (1, h_i)
		# This is used for gradient descent on the bias.
		DelBias = np.tensordot(DelLG, self.BiasDerivative, axes = 2) 
		
		# We need two derivatvies of the matrix multiplication. With respect to our weight matrix
		# (which will be used for updating gradient) and with respect to the input we got when feed-forwarding
		# (which will be piped down to the previous layer)
		self.matmul_derivative_wrtB(self.Forward_Input, self.WeightMat)
		self.matmul_derivative_wrtA(self.Forward_Input, self.WeightMat)

		# output: (1, h_i-1, h_i)
		# used for gradient update
		DelWeightMat = np.tensordot(DelLG, self.MatBDerivative, axes = 2)

		# in the first layer, we don't need to calculate this derivative anymore (there's no layer below)
		# so skip it to speed things up
		DelOut = None

		if not skipOutput:
			# output: (1, m, h_i-1)
			# this will be passed on to the previous layer
			DelOut = np.tensordot(DelLG, self.MatADerivative, axes = 2) 
		

		# do step in opposite direction of gradient
		self.Bias -= learningRate * DelBias[0]
		self.WeightMat -= learningRate * DelWeightMat[0]

		return DelOut


	def relu(self, mat):
		return np.maximum(0.0, mat)

	# uses the matrix mat of shape (m,n) and returns the tensor derivative of relu
	# of shape (m,n,m,n)
	def relu_derivative(self, mat):
		
		m, n = np.shape(mat)

		if self.ActivationDerivative is None:
			self.ActivationDerivative = np.zeros((m,n,m,n))
		else:
			self.ActivationDerivative.fill(0.0)

		for i in range(m):
			for j in range(n):
				if mat[i,j] > 0:
					self.ActivationDerivative[i,j,i,j] = 1.
		

	# given (m,k)-matrix A and (k, l)-matrix B, computes the derivative
	# of A*B (matrix multiplication) with respect to A
	# this will be a (m,l,m,k)-tensor
	def matmul_derivative_wrtA(self, A, B):
		m,k = np.shape(A)
		k,l = np.shape(B)

		if self.MatADerivative is None:
			self.MatADerivative = np.zeros((m,l,m,k))
		else:
			self.MatADerivative.fill(0.0)

		for a in range(m):
			self.MatADerivative[a,:,a,:] = B.transpose()

	# given (m,k)-matrix A and (k, l)-matrix B, computes the derivative
	# of A*B (matrix multiplication) with respect to A
	# this will be a (m,l,k,l)-tensor
	def matmul_derivative_wrtB(self, A, B):
		m,k = np.shape(A)
		k,l = np.shape(B)

		if self.MatBDerivative is None:
			self.MatBDerivative = np.zeros((m,l,k,l))
		else:
			self.MatBDerivative.fill(0.0)

		for b in range(l):
			self.MatBDerivative[:,b,:,b] = A

	# given matrix A, performs softmax over rows
	def softmax(self, A):

		# we use exp-normalization to avoid float overflows
		A = np.exp(A - np.amax(A, axis = 1))
		A = A / np.sum(A, axis = 1)
		return A

	# Calculate derivative of softmax that is performed over the rows of matrix A.
	# output will be (m,h_i,m,h_i)-tensor
	def softmax_derivative(self, A):
		m, n = np.shape(A)

		if self.ActivationDerivative is None:
			self.ActivationDerivative = np.zeros((m,n,m,n))
		else:
			self.ActivationDerivative.fill(0.0)

		softMat = self.softmax(A)

		for a in range(m):
			for b in range(n):
				for d in range(n):
					entry = 0.0 if b != d else softMat[a,b]

					# see notes
					row = np.copy(A[a,:])
					maxInRow = np.max(row)
					s = np.exp(row[0, b] + row[0, d] - 2*maxInRow)
					s /= (np.sum(np.exp(row - maxInRow))**2)

					entry = entry - s

					self.ActivationDerivative[a,b,a,d] = entry



# Load the MNIST dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


# Build up everything we need
train_Y = oneHotEncode(train_set[1])
test_Y = oneHotEncode(test_set[1])
train_X = np.matrix(train_set[0])
test_X = np.matrix(test_set[0])

train_X, test_X = normalize(train_X, test_X)

# learning rate
lr = 0.007
# number of learning iterations
numOfIters = 7000
# batch size for SGD
bs = 64

# Build up network
network = ClassificationNeuralNetwork(
	lr,
	train_X,
	train_Y,
	bs,
	numOfIters,
	decayLR = True
)

network.addHiddenLayer(20)
network.addHiddenLayer(20)
network.addOutputLayer()
network.learn()


# infer data and check how well we do on the train data
predictionMatrix = np.argmax(network.infer(train_X, train_Y), axis = 1)
trueLabels = np.argmax(train_Y, axis=1)
truePositives = (predictionMatrix == trueLabels).sum() * 1.0
print("Final accuracy on training data:" + str(truePositives / np.size(train_X,0)))

# infer data and check how well we do on the test data
predictionMatrix = np.argmax(network.infer(test_X, test_Y), axis = 1)
trueLabels = np.argmax(test_Y, axis=1)
truePositives = (predictionMatrix == trueLabels).sum() * 1.0
print("Final accuracy on test data:" + str(truePositives / np.size(test_X,0)))

