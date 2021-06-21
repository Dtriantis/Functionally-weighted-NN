from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
from sklearn.utils import shuffle
import tensorflow as tf

# Parameters
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.01
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps=1000, decay_rate =0.96, staircase = True)
# learning_rate = 0.01
training_epochs = 40000
batch_size = 200
display_step = 500

L=int(sys.argv[1])
M=200

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def init_weights(shape, name):
    """ Weight initialization """
    # weights = tf.random_normal(shape, stddev=0.01, mean =0)
    # return tf.Variable(weights,trainable=True, name = "weights")

    return tf.get_variable(shape = shape,name =  name, trainable=True,
         initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32, seed=42),
         regularizer=tf.contrib.layers.l2_regularizer(scale=0.00001))


def init_biases(shape):
    weights = tf.random_normal(shape, stddev=0.01, mean =0)
    return tf.Variable(weights,trainable=True, name = "biases")
#    return tf.Variable(tf.zeros(shape), trainable=True, name="biases")


def forwardprop1(X, x_size, L, M, name):

	#x_size = X.shape[1]   # Number of input nodes: 4 features and 1 bias
	# Weight initializations
	weights = init_weights(( x_size, L), name)
	biases = init_biases(( 1, L))

	s_array = np.zeros((L,M),dtype='float32')
	for i in xrange(L):
		for j in xrange(M):
			s_array[i][j] = np.power(np.cos(((2*i)-1/2.*M)-np.pi),j)

	s_array = tf.constant(s_array,name = "s_array")
	W_s = tf.matmul(weights, s_array,name = "W_S")

	B_s = tf.transpose(tf.matmul(biases, s_array,name = "B_S"))

	final_matrix = tf.add(tf.matmul(tf.transpose(W_s), tf.transpose(X),name = "W_X"),B_s,name = "W_X_B")

	return final_matrix

def forwardprop2(X, x_size, L, M, name):

	# Weight initializations
	weights_pol = init_weights(( x_size, L), name)	#diastash??

	s_array2 = np.zeros((L,M),dtype='float32')
	for i in xrange(L):
		for j in xrange(M):
			s_array2[i][j] = np.power(np.cos(((2*i)-1/2.*M)-np.pi),j)

	s_array2 = tf.constant(s_array2,name = "s_array2")
	W_s2 = tf.matmul(weights_pol, s_array2,name = "W_S")

	final_matrix2 = tf.matmul(W_s2, X)

	sum_2 = tf.reduce_sum(final_matrix2, 0, keep_dims=True,name = "reduce_sum_of_finalMatrix")
	sum_2 = tf.transpose(sum_2)

	pi_mi_ = np.divide(np.pi, M)
	pi_mi_.astype(np.float32)
	pi_mi = tf.constant(pi_mi_, dtype='float32')
	output = tf.scalar_mul(pi_mi, sum_2)

	return output

layer_1 = forwardprop1(x,n_input, L=L, M=M, name = 'l1')
activation = tf.nn.elu(layer_1)

output1 = forwardprop2(activation, 1, L=L, M=M, name = 'l12')
output2 = forwardprop2(activation, 1, L=L, M=M, name = 'l13')
output3 = forwardprop2(activation, 1, L=L, M=M, name = 'l14')
output4 = forwardprop2(activation, 1, L=L, M=M, name = 'l15')
output5 = forwardprop2(activation, 1, L=L, M=M, name = 'l16')
output6 = forwardprop2(activation, 1, L=L, M=M, name = 'l17')
output7 = forwardprop2(activation, 1, L=L, M=M, name = 'l18')
output8 = forwardprop2(activation, 1, L=L, M=M, name = 'l19')
output9 = forwardprop2(activation, 1, L=L, M=M, name = 'l10')
output0 = forwardprop2(activation, 1, L=L, M=M, name = 'l11')

#predict = tf.concat([output1, output2, output3], 1, name = "concatenation")
predict = tf.concat([output1, output2, output3, output4, output5, output6, output7, output8, output9,output0], 1, name = "concatenation")

prediction = tf.nn.softmax(predict)

# Define loss and optimizer
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))+sum(reg_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
print("L = " + str(L))
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
    test_accuracy = []

    for step in range(1, training_epochs+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x_test, batch_y_test = mnist.test.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            test_loss = sess.run([cost], feed_dict={x: batch_x_test, y: batch_y_test})

            # Calculate accuracy for 256 MNIST test images
            validation_accuracy = sess.run(accuracy, feed_dict={x: X_test[:3500],
                                      y: y_test[:3500]})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Acc= " + \
                  "{:.3f}".format(acc) + ", Validation Acc= " + \
                  "{:.3f}".format(validation_accuracy))

            losses.append(loss)
            validation_acc.append(validation_accuracy*100)
            test_accuracy.append(acc*100)
            test_cost.append(test_loss)
            steps.append(step)

    validation_acc.pop(0)
    test_cost.pop(0)
    test_accuracy.pop(0)
    steps.pop(0)
    losses.pop(0)

    print("Optimization Finished!")

    test_accur = sess.run(accuracy, feed_dict={x: X_test[3501:],y: y_test[3501:]})
    print("Testing Accuracy:",test_accur)

    plt.plot(steps, losses, '-b', label='train loss')
    plt.plot(steps, test_cost, '-r', label='validation set loss')
    plt.legend(loc='upper right')
    plt.title('L:' +str(L) +' total parameters: '+str(total_parameters) +' Testing Accuracy: '+str(test_accur))
    plt.grid(True)
    plt.savefig("/home/dtrianti/Desktop/fwnn/exp_plots/FWNN 1 layer 0001 regularization/"+str(L)+"validation_curves.png")

    plt.cla()
    plt.plot(steps, validation_acc, '-b', label='validation accuracy')
    plt.plot(steps, test_accuracy, '-r', label='train accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('batch step')
    plt.ylabel('validation set accuracy')
    plt.title('L1:' +str(L)+' total parameters: '+str(total_parameters) +' Testing Accuracy: '+str(test_accur))
    plt.grid(True)
    plt.savefig("/home/dtrianti/Desktop/fwnn/exp_plots/FWNN 1 layer 0001 regularization/loss"+str(L)+"val_acc.png")
