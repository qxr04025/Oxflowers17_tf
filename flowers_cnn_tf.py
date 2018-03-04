#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Oxford's 17 Category Flower Dataset classification task using a simple net.

Author:qinxiaoran
"""
from __future__ import division, print_function, absolute_import

import os
import os.path
from IPython import embed
import tensorflow as tf

import dataset
import create_batch

print(__doc__)

X_train, Y_train, X_val, Y_val = dataset.load_data(one_hot=True, resize_pics=(227, 227))
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 17])            
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tfconfig = tf.ConfigProto(allow_soft_placement = True)
tfconfig.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = tfconfig)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  

def conv2d(x, W, stride=[1, 2, 2, 1]):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_image = x
W_conv1 = weight_variable([5, 5, 3, 32])      
b_conv1 = bias_variable([32])       
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, stride=[1, 2, 2, 1]) + b_conv1)     
h_pool1 = max_pool(h_conv1)                                  

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, stride=[1, 2, 2, 1]) + b_conv2)      
h_pool2 = max_pool(h_conv2)

W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3, stride=[1, 1, 1, 1]) + b_conv3)      
h_pool3 = max_pool(h_conv3)

#W_conv4 = weight_variable([3, 3, 128, 256])
#b_conv4 = bias_variable([256])
#h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4, stride=[1, 1, 1, 1]) + b_conv4)      
#h_pool4 = max_pool(h_conv4)

shp = h_pool3.get_shape()
size = shp[1].value*shp[2].value*shp[3].value
h_pool3_flat = tf.reshape(h_pool3, [-1, size]) 

W_fc1 = weight_variable([size, 1024])
b_fc1 = bias_variable([1024])     
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32) 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  

W_fc2 = weight_variable([1024, 17])
b_fc2 = bias_variable([17])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices = 1))
train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

trainset = create_batch.create_trainset(X_train, Y_train)
for i in range(3000):
    batch = trainset.next_batch(64)
    _, train_loss = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i %20 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d,  train loss %g, train accuracy %g" % (i, train_loss, train_accuracy))
    if i % 200 == 0:
        val_accuracy = accuracy.eval(feed_dict={x: X_val, y_: Y_val, keep_prob: 1.0})
        val_loss = cross_entropy.eval(feed_dict={x: X_val, y_: Y_val, keep_prob: 1.0})
        print("step %d, val loss %g, val accuracy %g" % (i, val_loss, val_accuracy))
        print()


