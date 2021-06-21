from __future__ import print_function
import numpy as np
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import tensorflow as tf
import sys
import time
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Parameters
training_epochs = 2500
learning_rate = 0.005

batch_size = 100
display_step = 10

input_width = 28
input_height = 28

n_classes = 10 #total classes (0-9 digits)
M=200
# tf Graph input
x = tf.placeholder("float", [None, input_width*input_height])
y = tf.placeholder("float", [None, n_classes])

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.008, mean =0)
    return tf.Variable(weights,trainable=True, name = "weights")

def init_biases(shape):
    return tf.Variable(tf.zeros(shape), trainable=True, name="biases")

def cnn_1(X, L, M, kernel_height, kernel_width, input_width, input_height, s_array):

	# Weight initializations
	# weight is height and width of kernel
	# eg 5x5
	# and an extra L dimension

	weight_pad = input_height/2 - kernel_height/2

	weights = init_weights((kernel_height, kernel_width, L))
	biases = init_biases((1, 1, L))

	x_image = tf.reshape(X, [-1, input_width, input_height, 1])

	#reshape weights
	weights = tf.transpose(tf.reshape(weights, [kernel_height*kernel_width,L]))
	biases = tf.reshape(biases, [1,L])

	W_s = tf.matmul(tf.transpose(weights), s_array,name = "W_S")
	B_s = tf.matmul(biases, s_array,name = "B_S")

	#reshape again into height*weight
	W_s = tf.reshape(W_s, [kernel_height, kernel_width, 1, M])

	conv = tf.nn.conv2d(input = x_image,filter =  W_s, strides = [1, 1, 1, 1], padding='SAME')

	final_matrix = tf.add(conv, B_s)

	return final_matrix


def cnn_2(X, L, M, input_width, input_height, classes, s_array2):

	# Weight initializations

	#weights_pol = init_weights((input_width*input_height, L))
	weights_pol = init_weights((classes, input_width*input_height, L))

	W_s2 = tf.matmul(weights_pol, s_array2,name = "W_S")

	#reshape again into height*weight
	X = tf.reshape(X, [-1, input_width*input_height, M])

	W_s2 = tf.transpose(W_s2, perm=[2,0,1])
	X = tf.transpose(X, perm=[2,1,0])

	final_matrix2 = tf.matmul(W_s2,X)  #WITH 784 WEIGHT

	sum_2 = tf.reduce_sum(final_matrix2, 0, keep_dims=True, name = "reduce_sum_of_finalMatrix")
	sum_2 = tf.squeeze(sum_2, [0])

	pi_mi_ = np.divide(np.pi, M)
	pi_mi_.astype(np.float32)
	pi_mi = tf.constant(pi_mi_, dtype='float32')
	output = tf.scalar_mul(pi_mi, sum_2)

	output = tf.transpose(output)

	return output

L1=int(sys.argv[1])
L2=int(sys.argv[2])

s_array = np.zeros((L1,M),dtype='float32')
for i in xrange(L1):
	for j in xrange(M):
		s_array[i][j] = np.power(np.cos(((2*i)-1/2.*M)-np.pi),j)
s_array = tf.constant(s_array,name = "s_array")


s_array2 = np.zeros((n_classes, L2,M),dtype='float32')
for k in xrange(n_classes):
	for i in xrange(L2):
		for j in xrange(M):
			s_array2[k][i][j] = np.power(np.cos(((2*i)-1/2.*M)-np.pi),j)
s_array2 = tf.constant(s_array2,name = "s_array2")

#NETWORK
#network f_weighted CNN
layer_1 = cnn_1(x, L=L1, M=M, kernel_height=4, kernel_width=4, input_width=input_width, input_height=input_height, s_array=s_array)

#actication
activation = tf.nn.elu(layer_1)
#max pooling
activation = tf.layers.max_pooling2d(activation, 2, 2)
#integral layer
predict = cnn_2(activation, L=L2, M=M, input_width=14, input_height=14, classes=n_classes, s_array2=s_array2)
#softmax
prediction = tf.nn.softmax(predict)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('cost', loss_op)
tf.summary.scalar('acc', accuracy)

init = tf.global_variables_initializer()

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters

print("***************")
print("total parameters = " + str(total_parameters))
print("***************")
print("L1 = " + str(L1) + ",L2 = " + str(L2))
print("***************")
print("Learning rate = " + str(learning_rate))
print("***************")

X_test, y_test = shuffle(mnist.test.images[:], mnist.test.labels[:], random_state=0)

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    losses = []
    steps = []
    validation_acc = []
    test_cost = []

    writer = tf.summary.FileWriter("/home/dtrianti/Desktop/fwnn/logs/nn_logs", sess.graph)
    merged_summary_op = tf.summary.merge_all()

    for step in range(1, training_epochs+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x_test, batch_y_test = mnist.test.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc, summary = sess.run([loss_op, accuracy, merged_summary_op], feed_dict={x: batch_x, y: batch_y})
            test_loss = sess.run([loss_op], feed_dict={x: batch_x_test, y: batch_y_test})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

            # Calculate accuracy for 256 MNIST test images
            validation_accuracy = sess.run(accuracy, feed_dict={x: X_test[:3500],
                                      y: y_test[:3500]})

            losses.append(loss)
            validation_acc.append(validation_accuracy*100)
            test_cost.append(test_loss)

    print("Optimization Finished!")

    test_accur = sess.run(accuracy, feed_dict={x: X_test[3501:],y: y_test[3501:]})
    print("Testing Accuracy:",test_accur)

    plt.plot(steps, losses, '-b', label='train loss')
    plt.plot(steps, test_cost, '-r', label='validation set loss')
    plt.legend(loc='upper right')
    plt.title('L1:' +str(L1)+' L2:'+str(L2)+' total parameters: '+str(total_parameters) +' Testing Accuracy: '+str(test_accur))
    plt.grid(True)
    plt.savefig("/home/dtrianti/Desktop/fwnn/exp_plots/"+str(L1)+str(L2)+"validation_curves.png")

    plt.cla()
    plt.plot(steps, validation_acc)
    plt.xlabel('batch step')
    plt.ylabel('validation set accuracy')
    plt.title('L1:' +str(L1)+' L2:'+str(L2)+' total parameters: '+str(total_parameters) +' Testing Accuracy: '+str(test_accur))
    plt.grid(True)
    plt.savefig("/home/dtrianti/Desktop/fwnn/exp_plots/loss"+str(L1)+str(L2)+"val_acc.png")
