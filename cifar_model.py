import tensorflow as tf 
import numpy as np
import os

class cifar_model(object):
	"""
	A deep learning model in tensorflow to classify images in cifar-10. 
	"""

	def __init__(self, num_classes):
		self.num_classes = num_classes


	def add_placeholders(self):
		"""
		Add placeholders for input features and true labels
		"""
		self.input_x = tf.placeholder(tf.float32,[None,32,32,3], name = "input_x") 
		self.input_y = tf.placeholder(tf.float32,[None, self.num_classes], name = "input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

	def conv2d_relu(self, inputs, filter_h, filter_w, out_channels, module_name):
		"""
		Define a convolutional layer
		"""
		in_channel = inputs.get_shape().as_list()[-1]
		with tf.variable_scope(module_name):
			filter_shape = [filter_h,filter_w,in_channel,out_channels]
			
			W = tf.Variable(tf.truncated_normal(filter_shape,stddev = 0.1), name = "W")
			b = tf.Variable(tf.constant(0.1, shape = [out_channels]), name = "b")

			conv = tf.nn.conv2d(inputs,W,strides = [1,1,1,1], padding = "SAME", name = "conv")

			# Apply nonlinearity
			h = tf.nn.relu(tf.nn.bias_add(conv,b), name = "relu")

			return h

	def fc_layer(self, inputs, output_channel, module_name, last_layer = False):
		"""
		Define a fully connected layer
		"""
		input_channel = inputs.get_shape().as_list()[-1]
		with tf.variable_scope(module_name):
			b = tf.Variable(tf.constant(0.1, shape = [output_channel]), name = "b")
			W = tf.get_variable("W",shape = [input_channel,output_channel], initializer = tf.contrib.layers.xavier_initializer())
			

			if last_layer == True:
				h = tf.nn.xw_plus_b(inputs,W,b, name = "scores")
			else:
				h = tf.nn.xw_plus_b(inputs,W,b, name = "fc_out")

			return h

	def add_loss_op(self):
		"""
		Define Loss function

		"""
		with tf.variable_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
			self.loss = tf.reduce_mean(losses)

	def add_accuracy_op(self):
		"""
		Adding accuracy 
		"""
		with tf.variable_scope("accuracy"):
			correct_pred = tf.equal(self.predictions, tf.argmax(self.input_y,1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name = "accuracy")

	def build_graph(self):
		"""
		Build the CNN graph 
		"""
		self.add_placeholders()

		conv_out_1 = self.conv2d_relu(self.input_x, 3, 3, 48, "conv_relu_1")
		conv_out_2 = self.conv2d_relu(conv_out_1, 3, 3, 48, "conv_relu_2")

		self.pool_1 = tf.nn.max_pool(conv_out_2, ksize = [1, 2, 2, 1], strides = [1,1,1,1], padding = 'SAME', name = "max_pool")

		with tf.variable_scope("dropout"): 
			dout_1 = tf.nn.dropout(self.pool_1, self.dropout_keep_prob)
		with tf.variable_scope("Flatten"):
			flat_1 = tf.contrib.layers.flatten(dout_1)

		fc_out_1 = self.fc_layer(flat_1,512, module_name = "fc_1")
		fc_out_2 = self.fc_layer(fc_out_1, 256, module_name = "fc_2")

		self.scores = self.fc_layer(fc_out_2, self.num_classes, module_name = "output", last_layer = True)

		self.predictions = tf.argmax(self.scores, 1, name = "predictions")
		self.add_loss_op()
		self.add_accuracy_op()


	


			






















