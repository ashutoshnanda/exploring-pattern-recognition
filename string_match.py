#! /usr/local/bin/python3 -u

#Much thanks to the following references
# TensorFlow API
# "https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/multilayer_perceptron.py"

import numpy as np
import random
import tensorflow as tf

length_of_pattern = 10
length_of_strings = 20
number_of_train_examples = 10000
number_of_test_examples = 50000

def get_random_bits(length):
	return ''.join([('0' if random.random() < 0.5 else '1')\
				    for i in range(length)])

def insert_pattern(string, pattern):
	new_string = list(string)
	insert_index = random.choice(range(len(string) - len(pattern) + 1))
	new_string[insert_index:insert_index + len(pattern)] = list(pattern)
	return ''.join(new_string)

def examples_to_matrix(examples):
	matrix = [[(1 if char == '1' else 0) for j, char in enumerate(example)]\
	            for i, example in enumerate(examples)]
	return np.array(matrix)


def make_dataset(pattern_length, string_length, num_examples):
	hidden_pattern = get_random_bits(pattern_length)
	examples = [insert_pattern(get_random_bits(string_length), hidden_pattern)\
					 for i in range(num_examples // 2)]
	random_examples = [get_random_bits(string_length)\
	                   for i in range(num_examples // 2)]
	examples.extend(random_examples)
	fin_shape = (len(examples), 1)
	labels = np.array([(1 if hidden_pattern in example else 0)\
	                   for i, example in enumerate(examples)]).reshape(fin_shape)
	return (examples_to_matrix(examples), labels)

def do_neural_network(train_data, train_labels, test_data, test_labels):
	n_input = train_data.shape[1]
	n_hidden_1 = n_input
	n_hidden_2 = n_input
	n_classes = 1
	X = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, 1])
	weights = {
    	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    	#'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
	}
	biases = {
    	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    	#'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    	'out': tf.Variable(tf.random_normal([n_classes]))
	}
	predictions = neural_net_model(X, weights, biases)
	training_epochs = 3000
	learning_rate = 0.01
	#cost = tf.reduce_sum(-y * tf.log(predictions) - (1 - y) * tf.log(1 - predictions))
	cost = tf.reduce_sum(tf.square(predictions - y))
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
	init = tf.initialize_all_variables()
	threshold_predictions = tf.cast(tf.greater(predictions, 0.5), "float")
	correct_prediction = tf.equal(threshold_predictions, y)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	with tf.Session() as sess:
		sess.run(init)
		# Training cycle
		for epoch in range(training_epochs):
			sess.run(optimizer, feed_dict={X: train_data, y: train_labels})
			if epoch % 100 == 0:
				current_cost = sess.run(cost, feed_dict={X: train_data, y: train_labels})
				print("Epoch: %04d --> cost = %.9f" % (epoch + 1, current_cost))
				print("Accuracy: %.3f" % accuracy.eval({X: test_data, y: test_labels}))
				print("Accuracy: %.3f" % accuracy.eval({X: train_data, y: train_labels}))
		print("Optimization Finished!")
		print("=" * 80)
		print("Accuracy: %.3f" % accuracy.eval({X: test_data, y: test_labels}))
		print("Accuracy: %.3f" % accuracy.eval({X: train_data, y: train_labels}))

def neural_net_model(_X, _weights, _biases):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
	#layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
	#return tf.nn.sigmoid(tf.matmul(layer_2, _weights['out']) + _biases['out'])
	return tf.nn.sigmoid(tf.matmul(layer_1, _weights['out']) + _biases['out'])

if __name__ == '__main__':
	(train_data, train_labels) = make_dataset(length_of_pattern, length_of_strings, number_of_train_examples)
	(test_data, test_labels) = make_dataset(length_of_pattern, length_of_strings, number_of_test_examples)
	do_neural_network(train_data, train_labels, test_data, test_labels)
