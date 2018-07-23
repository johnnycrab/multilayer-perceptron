# Multilayer perceptron

My 'from scratch' implementation of a multilayer perceptron neural network. I did this to better understand how neural networks work and wanted to write a 'kerase'-style implementation where you can add arbitrary number of layers with arbitrary number of nodes. Unfortunately the more layers and the more nodes it slows down very quickly to unusable.

The implementation uses `numpy` and `cPickle` (for loading the testing data) and uses the MNIST handwritten digit dataset to test it on.

## Usage

Build up network using 

```python
network = ClassificationNeuralNetwork(
	0.007,			# learning Rate
	train_X,		# sample-feature matrix with samples in rows and features in columns
	train_Y, 		# one-hot-encoded matrix of labels
	64,				# batch size of samples used for one step of SGD
	7000,			# number of epochs
	decayLR = True	# if this is set to True then the learning rate will gradually decay
)
```

and add a number of layers using

```python
network.addHiddenLayer(20) # adds a hidden layer of dimension 20
network.addHiddenLayer(20) # adds another hidden layer of dimension 20
```

finish it off using 

```python
network.addOutputLayer()
network.learn()
```

Afterwards you can use `network.infer(train_X, train_Y)` to infer data. Here the first argument is a matrix of samples you want to classify. The second argument is the OHE-matrix of true labels which can be used directly in the `infer`-function to calculate the error.

For example using the setup above we get an accuracy of 95.35% on the test set.