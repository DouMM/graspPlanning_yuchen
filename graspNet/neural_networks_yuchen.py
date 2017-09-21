"""
Quick wrapper for grasp network
Author: yuchen, university of hamburg

reference: https://github.com/jireh-father/tensorflow-alexnet
"""

import copy
import json
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from autolab_core import YamlConfig
import ops as op

class graspCNN(object):
    """ Wrapper for grasp CNN """
    def __init__(self, config):
        """
        Parameters
        ----------
        config :obj: dict, python dictionary of configuration parameters
        """
        self._sess = None
        self._graph = tf.Graph()
        # load tensor params
        self._batch_size = config['batch_size']
        self._im_height = config['im_height']
        self._im_width = config['im_width']
        self._num_channels = config['im_channels']
        self._dropout_keep_prob = config['dropout_keep_prob']
        self.dim_grasp = config['dim_grasp']
        self.num_objClasses = config['num_objClasses']

    def initialize_network_alexnet(self):
        with tf.name_scope('inputlayer'):
            self.inputs = tf.placeholder("float", [self._batch_size, self._im_height, self._im_width, self._num_channels], 'inputs')
        self.dropout_keep_prob = tf.placeholder("float", None, 'keep_prob')
        # conv layer 1
        with tf.name_scope('conv1layer'):
            conv1, self.conv1_weight, self.conv1_baise = op.conv(self.inputs, 7, 96, 3)
            conv1 = op.lrn(conv1)
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
            tf.add_to_collection('weight', self.conv1_weight)
            tf.add_to_collection('weight', self.conv1_baise)
        # conv layer 2
        with tf.name_scope('conv2layer'):
            conv2, self.conv2_weight, self.conv2_baise = op.conv(conv1, 5, 256, 1, 1.0)
            conv2 = op.lrn(conv2)
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
            tf.add_to_collection('weight', self.conv2_weight)
            tf.add_to_collection('weight', self.conv2_baise)
        # conv layer 3
        with tf.name_scope('conv3layer'):
            conv3, self.conv3_weight, self.conv3_baise = op.conv(conv2, 3, 384, 1)
            tf.add_to_collection('weight', self.conv3_weight)
            tf.add_to_collection('weight', self.conv3_baise)
        # conv layer 4
        with tf.name_scope('conv4layer'):
            conv4, self.conv4_weight, self.conv4_baise = op.conv(conv3, 3, 384, 1, 1.0)
            tf.add_to_collection('weight', self.conv4_weight)
            tf.add_to_collection('weight', self.conv4_baise)
        # conv layer 5
        with tf.name_scope('conv5layer'):
            conv5, self.conv5_weight, self.conv5_baise = op.conv(conv4, 3, 256, 1, 1.0)
            conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            tf.add_to_collection('weight', self.conv5_weight)
            tf.add_to_collection('weight', self.conv5_baise)
        # fc layer 1
        with tf.name_scope('fc6layer'):
            fc6, self.fc6_weight, self.fc6_baise = op.fc(conv5, 4096, 1.0)
            fc6 = tf.nn.dropout(fc6, self.dropout_keep_prob)
            tf.add_to_collection('weight', self.fc6_weight)
            tf.add_to_collection('weight', self.fc6_baise)

        # object recognition (classfication problem)
        with tf.name_scope('fc7_object'):
            fc7_obj, self.fc7obj_weight, self.fc7obj_baise = op.fc(fc6, 4096, 1.0)
            fc7_obj = tf.nn.dropout(fc7_obj, self.dropout_keep_prob)
            tf.add_to_collection('obj_weight', self.fc7obj_weight)
            tf.add_to_collection('obj_weight', self.fc7obj_baise)
        with tf.name_scope('fc8_object'):
            self.output_obj, self.fc8obj_weight, self.fc8obj_baise = op.fc(fc7_obj, self.num_objClasses, 1.0, None)
            tf.add_to_collection('obj_weight', self.fc8obj_weight)
            tf.add_to_collection('obj_weight', self.fc8obj_baise)
            #self.output_obj_softmax = tf.softmax(self.output_obj)
        #--- grasp pose (regression problem)
        with tf.name_scope('fc7_grasp'):
            fc7_grasp, self.fc7grasp_weight, self.fc7grasp_baise = op.fc(fc6, 4096, 1.0)
            fc7_grasp = tf.nn.dropout(fc7_grasp, self.dropout_keep_prob)
            tf.add_to_collection('grasp_weight', self.fc7grasp_weight)
            tf.add_to_collection('grasp_weight', self.fc7grasp_baise)
        with tf.name_scope('fc8_grasp'):
            self.output_grasp, self.fc8grasp_weight, self.fc8grasp_baise = op.fc(fc7_grasp, self.dim_grasp, 1.0, None)
            tf.add_to_collection('grasp_weight', self.fc8grasp_weight)
            tf.add_to_collection('grasp_weight', self.fc8grasp_baise)

        #---mask recognition (regression problem)
        with tf.name_scope('fc8_mask'):
            input_shape = fc7_obj.get_shape().as_list()
            self.fc8mask_weight = tf.Variable(tf.random_normal([input_shape[-1], 4096], dtype=tf.float32, stddev=0.01),
                                 name='weights')
            self.fc8mask_biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), name='biases')
            fc8_mask = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc7_obj + fc7_grasp, self.fc8mask_weight), self.fc8mask_biases))
            tf.add_to_collection('mask_weight', self.fc8mask_weight)
            tf.add_to_collection('mask_weight', self.fc8mask_biases)

        with tf.name_scope('fc9_mask'):
            self.output_mask, self.fc9mask_weight, self.fc9mask_baise = op.fc(fc8_mask, self._im_height*self._im_width, 1.0, None)
            tf.add_to_collection('mask_weight', self.fc9mask_weight)
            tf.add_to_collection('mask_weight', self.fc9mask_baise)

        with tf.name_scope('loss'):
            with tf.name_scope('OBJECT_crossEntropy'):
                self.labels_object = tf.placeholder(shape=[self._batch_size, self.num_objClasses],dtype=tf.int32, name='object_labels')
                # classfic.
                self.loss_obj = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_obj,
                                                            labels=self.labels_object, name='cross-entropy'))
            with tf.name_scope('GRASP_squareError'):
                self.labels_grasp = tf.placeholder(shape=[self._batch_size, self.dim_grasp],dtype=tf.float32, name='grasp_labels')
                # regression.
                self.loss_grasp = tf.reduce_mean(tf.square(self.output_grasp-self.labels_grasp))
            with tf.name_scope('Mask_squareError'):
                self.labels_mask = tf.placeholder(shape=[self._batch_size, self._im_height*self._im_width], dtype=tf.float32, name='grasp_labels')
                # regression.
                self.loss_mask = tf.reduce_mean(tf.square(self.output_mask - self.labels_mask))
            with tf.name_scope('loss_l2'):
                lmbda = 5e-04
                self.l2_loss = tf.reduce_sum(lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
            self.loss_all = self.loss_grasp
    def save_scalar(self, step, name, value, writer):
        """Save a scalar value to tensorboard.
          Parameters
          ----------
          step: int
            Training step (sets the position on x-axis of tensorboard graph.
          name: str
            Name of variable. Will be the name of the graph in tensorboard.
          value: float
            The value of the variable at this step.
          writer: tf.FileWriter
            The tensorboard FileWriter instance.
          """
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = float(value)
        summary_value.tag = name
        writer.add_summary(summary, step)
#----------------------------------------
    '''
    def initialize_network(self, myScope):
        self._input_im_node =  tf.placeholder(shape=[64,32, 32, 1],dtype=tf.float32, name='imageIn') #imageIn: a tensor of size [batch_size, height, width, 1]
        self.conv1 = tf.contrib.layers.convolution2d( \
                        inputs=self._input_im_node,num_outputs=32,\
                        kernel_size=[8,8],stride=[4,4],padding='VALID', \
                        activation_fn=tf.nn.relu, biases_initializer=None, reuse=None, scope=myScope+'_conv1')
        self.conv2 = tf.contrib.layers.convolution2d( \
                        inputs=self.conv1,num_outputs=64,\
                        kernel_size=[4,4],stride=[2,2],padding='VALID', \
                        activation_fn=tf.nn.relu, biases_initializer=None, reuse=None, scope=myScope+'_conv2')
        self.conv3Flat = tf.contrib.layers.flatten(self.conv2)

        # 2. task.
        self.fc1 = tf.contrib.layers.fully_connected(self.conv3Flat, num_outputs=1024,
                                                            activation_fn=tf.nn.relu, reuse=None,scope=myScope+'_fc1_source')
        self.fc2 = tf.contrib.layers.fully_connected(self.fc1, num_outputs=512,
                                                            activation_fn=tf.nn.relu, reuse=None,scope=myScope+'_fc2_source')
        self._output_tensor = tf.contrib.layers.fully_connected(self.fc2, 4,\
                                                            activation_fn=None, reuse=None, scope=myScope+'_output_tensor')

        #Loss function:
        self.labels = tf.placeholder(shape=[64, 4],dtype=tf.float32, name=myScope+'_output_poses_node')
        with tf.name_scope(myScope+'_loss'):
            self.loss_predict = tf.nn.l2_loss(tf.subtract(self._output_tensor, self.labels))
    '''
