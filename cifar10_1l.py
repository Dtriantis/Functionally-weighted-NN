from __future__ import division, print_function, absolute_import
#mechanism to dynamically include the relative path where utils.ipynb is housed to the module search path.
from inspect import getsourcefile
import os
import os.path
import sys
import time
import re
import cPickle
import urllib, tarfile

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n_classes=10
batch_size=256
image_width=32
image_height=32
image_depth=3
learning_rate=0.01
n_epochs=5000
n_validate_samples=2500
n_test_samples=5
n_checkpoint_steps=5

L1=12
L2=11
M=150

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
def maybe_download_and_extract(dest_directory):
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
            float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    with tarfile.open(filepath, 'r:gz') as t:
        dataset_dir = os.path.join(dest_directory, t.getmembers()[0].name)
        t.extractall(dest_directory)

    return dataset_dir

dataset_dir = maybe_download_and_extract('./../data')
print(dataset_dir)

checkpoint_dir = os.path.join(dataset_dir, 'chkpt_cifar10')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

#regular expression that matches a datafile
r_data_file = re.compile('^data_batch_\d+')
#training and validate datasets as numpy n-d arrays,
#apropriate portions of which are ready to be fed to the placeholder variables
train_all={'data':[], 'labels':[]}
validate_all={'data':[], 'labels':[]}
test_all={'data':{}, 'labels':[]}
label_names_for_validation_and_test=None

def unpickle(relpath):
    with open(relpath, 'rb') as fp:
        d = cPickle.load(fp)
    return d

def prepare_input(data=None, labels=None):
    global image_height, image_width, image_depth
    assert(data.shape[1] == image_height * image_width * image_depth)
    assert(data.shape[0] == labels.shape[0])
    #do mean normaization across all samples
    mu = np.mean(data, axis=0)
    mu = mu.reshape(1,-1)
    sigma = np.std(data, axis=0)
    sigma = sigma.reshape(1, -1)
    data = data - mu
    data = data / sigma
    is_nan = np.isnan(data)
    is_inf = np.isinf(data)
    if np.any(is_nan) or np.any(is_inf):
        print('data is not well-formed : is_nan {n}, is_inf: {i}'.format(n= np.any(is_nan), i=np.any(is_inf)))
    #data is transformed from (no_of_samples, 3072) to (no_of_samples , image_height, image_width, image_depth)
    #make sure the type of the data is no.float32
    data = data.reshape([-1,image_depth, image_height, image_width])
    data = data.transpose([0, 2, 3, 1])
    data = data.astype(np.float32)

    return data, labels

def convert_to_rgb_img_data(index=-1, data=None):
    assert(index < data.shape[0])
    image_holder = np.zeros(shape=[data.shape[1],data.shape[2], data.shape[3]], dtype=np.float32)
    image_holder[:, :, :] = data[index, :, :, :]
    plt.imshow(image_holder)

def load_and_preprocess_input(dataset_dir=None):
    assert(os.path.isdir(dataset_dir))
    global train_all, validate_all, label_names_for_validation_and_test
    trn_all_data=[]
    trn_all_labels=[]
    vldte_all_data=[]
    vldte_all_labels=[]
    tst_all_data=[]
    tst_all_labels=[]
    #for loading train dataset, iterate through the directory to get matchig data file
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            m=r_data_file.match(f)
            if m:
                relpath = os.path.join(root, f)
                d=unpickle(os.path.join(root, f))
                trn_all_data.append(d['data'])
                trn_all_labels.append(d['labels'])
    #concatenate all the  data in various files into one ndarray of shape
    #data.shape == (no_of_samples, 3072), where 3072=image_depth x image_height x image_width
    #labels.shape== (no_of_samples)
    trn_all_data, trn_all_labels = (np.concatenate(trn_all_data).astype(np.float32),
                                          np.concatenate(trn_all_labels).astype(np.int32))

    #load the only test data set for validation and testing
    #use only the first n_validate_samples samples for validating
    test_temp=unpickle(os.path.join(dataset_dir, 'test_batch'))
    vldte_all_data=test_temp['data'][0:(n_validate_samples+n_test_samples), :]
    vldte_all_labels=test_temp['labels'][0:(n_validate_samples+n_test_samples)]
    vldte_all_data, vldte_all_labels =  (np.concatenate([vldte_all_data]).astype(np.float32),
                                             np.concatenate([vldte_all_labels]).astype(np.int32))
     #transform the test images in the same manner as the train images
    train_all['data'], train_all['labels'] = prepare_input(data=trn_all_data, labels=trn_all_labels)
    validate_and_test_data, validate_and_test_labels = prepare_input(data=vldte_all_data, labels=vldte_all_labels)

    validate_all['data'] = validate_and_test_data[0:n_validate_samples, :, :, :]
    validate_all['labels'] = validate_and_test_labels[0:n_validate_samples]
    test_all['data'] = validate_and_test_data[n_validate_samples:(n_validate_samples+n_test_samples), :, :, :]
    test_all['labels'] = validate_and_test_labels[n_validate_samples:(n_validate_samples+n_test_samples)]

    #load all label-names
    label_names_for_validation_and_test=unpickle(os.path.join(dataset_dir, 'batches.meta'))['label_names']

load_and_preprocess_input(dataset_dir=dataset_dir)

def init_weights(shape, namer):
    """ Weight initialization """
    return tf.get_variable(shape = shape,name =  namer, trainable=True,
        initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32, seed=42),
        regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))

def init_biases(shape, namer):
    return tf.Variable(tf.zeros(shape), trainable=True, name=namer)

input_chanels = 3

s_array = np.zeros((input_chanels,L1,M),dtype='float32')
for k in xrange(input_chanels):
    for i in xrange(L1):
        for j in xrange(M):
            s_array[k][i][j] = np.power(np.cos(((2*i)-1/2.*M)-np.pi),j)
s_array = tf.constant(s_array,name = "s_array")

s_array_b = np.zeros((L1,M),dtype='float32')
for i in xrange(L1):
    for j in xrange(M):
        s_array_b[i][j] = np.power(np.cos(((2*i)-1/2.*M)-np.pi),j)
s_array_b = tf.constant(s_array_b,name = "s_array_b")

s_array2 = np.zeros((M, L1, M),dtype='float32')
for k in xrange(M):
    for i in xrange(L1):
        for j in xrange(M):
            s_array2[k][i][j] = np.power(np.cos(((2*i)-1/2.*M)-np.pi),j)
s_array2 = tf.constant(s_array2,name = "s_array2")

s_array3 = np.zeros((n_classes, L2,M),dtype='float32')
for k in xrange(n_classes):
    for i in xrange(L2):
        for j in xrange(M):
            s_array3[k][i][j] = np.power(np.cos(((2*i)-1/2.*M)-np.pi),j)
s_array3 = tf.constant(s_array3,name = "s_array3")

def cnn_1_1(X, L, M, kernel_height, kernel_width, input_width, input_height, s_array, s_array_b, w_name, b_name):

    weights2 = init_weights((kernel_height, kernel_width, L, 3), namer=w_name)
    biases = init_biases((1, L), namer=b_name)

    x_image = tf.reshape(X, [-1, input_width, input_height, 3])

    #reshape weights
    weights2 = tf.reshape(weights2, [kernel_height*kernel_width,L, 3])
    weights2 = tf.transpose(weights2, perm=[1,0,2])
    s_array = tf.transpose(s_array, perm=[0,1,2])

    W_s = tf.matmul(tf.transpose(weights2), s_array,name = "W_S1")

    s_array = tf.transpose(s_array, perm=[1,0,2])

    B_s = tf.matmul(biases, s_array_b,name = "B_S1")
    #reshape again into height*weight
    W_s = tf.reshape(W_s, [kernel_height, kernel_width, input_chanels, M])

    conv = tf.nn.conv2d(input = x_image,filter =  W_s, strides = [1, 1, 1, 1], padding='SAME')

    final_matrix = tf.add(conv, B_s)

    return final_matrix

def cnn_1_2(X, L, M, kernel_height, kernel_width, input_width, input_height, s_array2, s_array_b, w_name, b_name):

    weights2 = init_weights((kernel_height, kernel_width, L, M), namer=w_name)
    biases = init_biases((1, L), namer = b_name)
    x_image = tf.reshape(X, [-1, input_width, input_height, M])

    #reshape weights
    weights2 = tf.reshape(weights2, [kernel_height*kernel_width,L,M])
    weights2 = tf.transpose(weights2, perm=[1,0,2])
    s_array2 = tf.transpose(s_array2, perm=[0,1,2])

    W_s = tf.matmul(tf.transpose(weights2), s_array2,name = "W_S1")
    B_s = tf.matmul(biases, s_array_b,name = "B_S2")

    #reshape again into height*weight
    W_s = tf.reshape(W_s, [kernel_height, kernel_width, M, M])

    conv = tf.nn.conv2d(input = x_image,filter =  W_s, strides = [1, 1, 1, 1], padding='SAME')

    biases = tf.reshape(biases, [1,1,L])

    final_matrix = tf.add(conv, B_s)

    return final_matrix

def cnn_2(X, L, M, input_width, input_height, classes, s_array3, w_name, b_name):

    # Weight initializations
    weights_pol = init_weights((classes, input_width*input_height, L), namer=w_name)

    W_s2 = tf.matmul(weights_pol, s_array3,name = "W_S")

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

def wireup(x, batch_size=256, n_classes=10, scope='dummy'):
    with tf.variable_scope(scope) as sc:

        #network f_weighted CNN
        layer_0 = cnn_1_1(x, L=L1, M=M, kernel_height=3, kernel_width=3, input_width=32, input_height=32, s_array=s_array,s_array_b=s_array_b, w_name='L0', b_name='b0')
        activation0 = tf.nn.relu(layer_0)

        layer_1 = cnn_1_2(activation0, L=L1, M=M, kernel_height=3, kernel_width=3, input_width=32, input_height=32, s_array2=s_array2,s_array_b=s_array_b, w_name='L1', b_name='b1')
        activation1 = tf.nn.relu(layer_1)

        pool1 = tf.layers.max_pooling2d(activation1, 2, 2)

        layer_3 = cnn_1_2(pool1, L=L1, M=M, kernel_height=3, kernel_width=3, input_width=16, input_height=16, s_array2=s_array2,s_array_b=s_array_b, w_name='L2', b_name='b2')
        activation3 = tf.nn.relu(layer_3)

        layer_4 = cnn_1_2(activation3, L=L1, M=M, kernel_height=3, kernel_width=3, input_width=16, input_height=16, s_array2=s_array2,s_array_b=s_array_b, w_name='L3', b_name='b3')
        activation4 = tf.nn.relu(layer_4)

        pool2 = tf.layers.max_pooling2d(activation4, 2, 2)

        layer_6 = cnn_1_2(pool2, L=L1, M=M, kernel_height=3, kernel_width=3, input_width=8, input_height=8, s_array2=s_array2,s_array_b=s_array_b, w_name='L4', b_name='b4')
        activation6 = tf.nn.relu(layer_6)

        layer_7 = cnn_1_2(activation6, L=L1, M=M, kernel_height=3, kernel_width=3, input_width=8, input_height=8, s_array2=s_array2,s_array_b=s_array_b, w_name='L5', b_name='b5')
        activation7 = tf.nn.relu(layer_7)

        #integral layer
        logits = cnn_2(activation7, L=L2, M=M, input_width=8, input_height=8, classes=n_classes, s_array3=s_array3, w_name='L6', b_name='b6')

    return logits

def compute_loss(logits, labels):

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # reg_constant = 0.001  # Choose an appropriate one.

    print('compute_loss: {lg}, {lb}'.format(lg=logits.get_shape(), lb=labels.get_shape()))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example') + sum(reg_losses)

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean

X = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, image_depth))
y = tf.placeholder(tf.float32, shape=(batch_size, 10))

global_step = tf.Variable(0, trainable=False)
logits=wireup(X, batch_size=batch_size, n_classes=10, scope='train')
loss = compute_loss(logits, y)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step)

train_prediction = tf.nn.softmax(logits)
accuracy_1 = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(y, 1))
accuracy=tf.reduce_mean(tf.cast(accuracy_1,tf.float32),0)
_, top_k_pred = tf.nn.top_k(logits, k=5)
init = tf.initialize_all_variables()
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
n_batches_train = int(train_all['labels'].shape[0]//batch_size)
n_batches_validate = int(validate_all['labels'].shape[0]//batch_size)

def checkpoint(g):
    global dataset_dir, checkpoint_dir
    dataset_name = os.path.split(dataset_dir)[1]
    if not os.path.isdir(os.path.join(dataset_dir, 'out')):
        os.makedirs(os.path.join(dataset_dir, 'out'))
    loss_fig_path= os.path.join(dataset_dir, 'out', '%s_train_loss.png' % dataset_name)
    acc_fig_path= os.path.join(dataset_dir, 'out', '%s_acc.png' % dataset_name)
    saver.save(sess, os.path.join(checkpoint_dir, 'train_cifar10.ckpt'),
                       global_step=g)

def all_batches_run_train(n_batches, data=None, labels=None):
    sum_all_batches_loss =0
    sum_all_batches_acc=0
    sum_n_samples=0
    for b in xrange(n_batches):
            offset = b * batch_size
            batch_data = data[offset : offset+batch_size, :, :, :]
            n_samples = batch_data.shape[0]
            batch_labels = labels[offset: offset+batch_size]
            batch_labels = (np.arange(n_classes) == batch_labels[:, None]).astype(np.float32)
            feed_dict = {X: batch_data,
                         y: batch_labels}
            _, loss_value, a =sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)
            sum_all_batches_loss += loss_value * n_samples
            sum_all_batches_acc += a * n_samples
            sum_n_samples += n_samples
            if(n_samples != batch_size):
                print('n_samples =%d' % n_samples)
    return (sum_all_batches_loss/sum_n_samples, sum_all_batches_acc/sum_n_samples)

def all_batches_run_validate(n_batches, data=None, labels=None):
    sum_all_batches_acc=0
    sum_n_samples=0
    for b in xrange(n_batches):
            offset = b * batch_size
            batch_data = data[offset : offset+batch_size, :, :, :]
            n_samples = batch_data.shape[0]
            batch_labels = labels[offset: offset+batch_size]
            batch_labels = (np.arange(n_classes) == batch_labels[:, None]).astype(np.float32)
            feed_dict = {X: batch_data,
                         y: batch_labels}
            a, top_k = sess.run([accuracy, top_k_pred], feed_dict=feed_dict)
            sum_all_batches_acc += a * n_samples
            sum_n_samples += n_samples
            if(n_samples != batch_size):
                print('n_samples =%d' % n_samples)

    return sum_all_batches_acc/sum_n_samples

def run_test(data=None, labels=None):
    assert(data.shape[0] == labels.shape[0])
    batch_data = np.zeros(shape=(batch_size, data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)
    batch_labels= np.zeros(shape=(batch_size), dtype=np.int32)
    batch_data[0:data.shape[0], : , :, :] = data[:, :, :, :]
    batch_labels[0:data.shape[0],] = labels[:]
    batch_labels = (np.arange(n_classes) == batch_labels[:, None]).astype(np.float32)
    feed_dict = {X: batch_data, y: batch_labels}
    a, top_k = sess.run([accuracy, top_k_pred], feed_dict=feed_dict)
    print('testing %r images' % n_test_samples)
    for i in range(n_test_samples):
        print('test_image: %s' % label_names_for_validation_and_test[test_all['labels'][i]],
              ' top-5 matches %r'%[ label_names_for_validation_and_test[j] for j in top_k[i]])

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

with tf.Session() as sess:
    sess.run(init)
    print('starting training')

    losses = []
    steps = []
    validation_acc = []
    train_accuracy = []

    for e in xrange(n_epochs):

        start_time = time.time()

        n_data = train_all['data'].shape[0]

        perm = np.random.permutation(n_data)
        permuted_data = train_all['data'][perm,:, :, :]
        permuted_labels = train_all['labels'][perm]

        validate_data = validate_all['data']
        validate_labels = validate_all['labels']

        mean_loss_per_sample_train, accuracy_per_sample_train =all_batches_run_train(n_batches_train, data=permuted_data, labels=permuted_labels)
        accuracy_per_sample_validate=all_batches_run_validate(n_batches_validate, data=validate_data, labels=validate_labels)

        losses.append(mean_loss_per_sample_train)
        validation_acc.append(accuracy_per_sample_validate*100)
        train_accuracy.append(accuracy_per_sample_train*100)
        steps.append(e)

        duration = time.time() - start_time
        if ((e+1)% n_checkpoint_steps) == 0:
            print('loss for global_step {g}, epoch {e}, train_loss={l},  validate-accuracy={a},  time={t}'.format(g=global_step.eval(), e=e, l=mean_loss_per_sample_train, a=accuracy_per_sample_validate, t=duration))
            checkpoint(global_step.eval())
    print('done training')
    run_test(data=test_all['data'], labels=test_all['labels'])
    print('done testing')

    plt.plot(steps, losses, '-b', label='train loss')
    plt.title('total parameters: '+str(total_parameters))
    plt.grid(True)
    plt.savefig("/home/dtrianti/Desktop/results_cifar/"+str(L1)+str(L2)+"validation_curves.png")

    plt.cla()
    plt.plot(steps, validation_acc, '-b', label='validation accuracy')
    plt.plot(steps, train_accuracy, '-r', label='train accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('step')
    plt.ylabel('accuracy')
    plt.title('total parameters: '+str(total_parameters))
    plt.grid(True)
    plt.savefig("/home/dtrianti/Desktop/results_cifar/"+str(L1)+str(L2)+"val_acc.png")
