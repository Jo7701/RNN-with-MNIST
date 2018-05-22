import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

rnn_size = 1024
n_chunks = 28
chunk_size = 28
n_classes = 10
batch_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size], name='x')
y = tf.placeholder('float', name = 'y')

def neural_network(x):
	layer = {'weights' : tf.Variable(tf.random_normal([rnn_size, n_classes])), 'biases' : tf.Variable(tf.random_normal([n_classes]))}

	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, chunk_size])
	x = tf.split(x, n_chunks)
	
	lstm = rnn_cell.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm, x, dtype = tf.float32)

	output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'], name = 'output')

	return output


def train(x):
	prediction = neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 5
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples / batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape([batch_size, n_chunks, chunk_size])
				_, c  = sess.run([optimizer, cost], feed_dict={x: epoch_x, y:epoch_y})
				epoch_loss += c
			print 'Epoch ', epoch + 1, ' Loss: ', epoch_loss
			os.makedirs('Epoch'+str(epoch+1))
			saver.save(sess, 'Epoch'+str(epoch+1)+'/model')

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy: ', accuracy.eval({x:mnist.test.images.reshape(-1, n_chunks, chunk_size), y:mnist.test.labels})

train(x)
