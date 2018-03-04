#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" 
Oxford's 17 Category Flower Dataset classification task using Alexnet.

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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tfconfig = tf.ConfigProto(allow_soft_placement = True)
tfconfig.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = tfconfig)

fid_train = open('acc_train.txt', 'w')
fid_val = open('acc_val.txt', 'w')

def conv2d(input_op, shape, stride, padding, name, p):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1), name='weights')
        b = tf.Variable(tf.constant(0.1, shape=[shape[-1]], dtype=tf.float32), trainable = True, name='biases')
        conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_op, W, strides=stride, padding=padding), b), name=scope)
        p += [tf.reduce_max(tf.abs(W)), b]
        return conv

def maxpool2d(input_op, shape, stride, padding, name):
    pool = tf.nn.max_pool(input_op, ksize=shape, strides=stride, padding=padding, name=name)
    return pool

def lrn2d(input_op, name):
    lrn = tf.nn.local_response_normalization(input_op, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name=name)
    return lrn

def fc(input_op, shape, name, p):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1), name='weights')
        b = tf.Variable(tf.constant(0.1, shape=[shape[-1]], dtype=tf.float32), name='biases')
        fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(input_op, W), b), name=scope)
        p += [tf.reduce_max(tf.abs(W)), b]
        return fc

def softmax(input_op, shape, name, p):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1), name='weights')
        b = tf.Variable(tf.constant(0.1, shape=[shape[-1]], dtype=tf.float32), name='biases')
        sf = tf.nn.softmax(tf.nn.bias_add(tf.matmul(input_op, W), b), name=scope)
        p += [tf.reduce_max(tf.abs(W)), b]
        return sf

def dropout(input_op, keep_prob, name):
    drop = tf.nn.dropout(input_op, keep_prob, name=name)
    return drop

def inference(x, keep_prob):
#x = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
#y_ = tf.placeholder(tf.float32, shape=[None, 17])
    p = []

    conv1 = conv2d(x, [11, 11, 3, 96], [1, 4, 4, 1], padding='SAME', name='conv1', p=p)
    #conv1_shp = conv1.get_shape()
    #print(conv1_shp[0].value, " ", conv1_shp[1].value, " ", conv1_shp[2].value, " ", conv1_shp[3].value)
    pool1 = maxpool2d(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='pool1')
    lrn1 = lrn2d(pool1, name='lrn1')
    conv2 = conv2d(lrn1, [5, 5, 96, 256], [1, 1, 1, 1], padding='SAME', name='conv2', p=p)
    pool2 = maxpool2d(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='pool2')
    lrn2 = lrn2d(pool2, name='lrn2')
    conv3 = conv2d(lrn2, [3, 3, 256, 384], [1, 1, 1, 1], padding='SAME', name='conv3', p=p)
    conv4 = conv2d(conv3, [3, 3, 384, 384], [1, 1, 1, 1], padding='SAME', name='conv4', p=p)
    conv5 = conv2d(conv4, [3, 3, 384, 256], [1, 1, 1, 1], padding='SAME', name='conv5', p=p)
    pool5 = maxpool2d(conv5, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='pool5')
    lrn5 = lrn2d(pool5, name='lrn5')

    shp = lrn5.get_shape()
    flatted_shape = shp[1].value*shp[2].value*shp[3].value
    lrn5_reshp = tf.reshape(lrn5, [-1, flatted_shape], name='lrn5_reshp')

    fc6 = fc(lrn5_reshp, [flatted_shape, 4096], name='fc6', p=p)
    fc6_drop = dropout(fc6, keep_prob, name='fc6_drop')
    fc7 = fc(fc6_drop, [4096, 4096], name='fc7', p=p)
    fc7_drop = dropout(fc7, keep_prob, name='fc7_drop')
    fc8 = softmax(fc7_drop, [4096, 17], name='fc8', p=p)
    y_conv = fc8
    return y_conv, p

x = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 17])
keep_prob = tf.placeholder(tf.float32)
y_conv, p = inference(x, keep_prob)
# X_batch, Y_batch = tf.train.shuffle_batch([X_train, Y_train], batch_size=64, capacity=80+3*64, min_after_dequeue=80, enqueue_many=True)
# y_conv = inference(X_batch, keep_prob)
# y_ = tf.cast(Y_batch, tf.float32)
#y_conv /= tf.reduce_sum(y_conv, reduction_indices=1, keep_dims=True)
#loss = tf.reduce_mean(tf.square(tf.subtract(y_,y_conv)))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices = 1))
train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()

#tf.train.start_queue_runners(sess=sess)

trainset = create_batch.create_trainset(X_train, Y_train)
for i in range(20000):
    batch = trainset.next_batch(64)
#    X_batch, Y_batch = tf.train.shuffle_batch([X_train, Y_train], batch_size=64, capacity=80+3*64, min_after_dequeue=80, enqueue_many=True)
#    train_step.run(feed_dict={x: X_batch, y_: Y_batch, keep_prob: 0.5})
    _, train_loss, para = sess.run([train_step, cross_entropy, p], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i %20 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d,  train loss %g, train accuracy %g" % (i, train_loss, train_accuracy))
        #print(para)
        if i % 200 == 0:
            fid_train.write(str(i) + ' ' + str(train_accuracy) + ' ' + str(train_loss) + '\n')
    if i % 200 == 0:
        val_accuracy = accuracy.eval(feed_dict={x: X_val, y_: Y_val, keep_prob: 1.0})
        val_loss = cross_entropy.eval(feed_dict={x: X_val, y_: Y_val, keep_prob: 1.0})
        fid_val.write(str(i) + ' ' + str(val_accuracy) + ' ' + str(val_loss) + '\n')
        print("step %d, val loss %g, val accuracy %g" % (i, val_loss, val_accuracy))
        print()

fid_train.close()
fid_val.close()