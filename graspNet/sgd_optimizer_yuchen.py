"""
Optimizer class for training a graspcnn object.
Author: yuchen
"""
#!/usr/bin/env python
# coding=utf-8
import copy
import json
import logging
import numpy as np
import os
import sys
import shutil
import time
import tensorflow as tf
import autolab_core.utils as utils
import collections

from graspNet.learning_analysis_yuchen import ClassificationResult, RegressionResult
from graspNet.optimizer_constants_yuchen import ImageMode, TrainingMode, PreprocMode, InputDataMode, GeneralConstants, ImageFileTemplates
from graspNet.train_stats_logger_yuchen import TrainStatsLogger

# general optimization parameters
train_batch_size: 64
val_batch_size: 64

data_split_mode: image_wise # how to split up the data into training vs validation: options are image_wise, stable_pose_wise, object_wise
train_pct: 0.8 # percentage of the data to use for training vs validation
total_pct: 1.0 # percentage of all the files to use
eval_total_train_error: 0

loss: l2
optimizer: rmsprop
base_lr: 0.001
decay_step_multiplier: 0.8   # number of times to go through training datapoints before stepping down decay rate
decay_rate: 0.96
momentum_rate: 0.9

### GQCNN CONFIG ###
graspcnn_config:
  # basic data metrics
  im_height: 50
  im_width: 50
  im_channels: 3
  # needs to match input data mode that was used for training, determines the pose dimensions for the network
  input_data_mode: yuchen_pose_mode
        self.graspcnn.initialize_network_alexnet()

        # 2. create optimizer
        global_step = tf.Variable(0)
        self.decay_step = self.decay_step_multiplier * self.num_train # use 0.66
        self.learning_rate = tf.train.exponential_decay(self.base_lr,                # base learning rate.
            											global_step * self.train_batch_size,  # current index into the dataset.
            											self.decay_step,          # decay step.
            											self.decay_rate,          # decay rate.
            											staircase=True)

        with tf.name_scope('optimizer_obj'):
            self.varList_obj = tf.get_collection('weight') + tf.get_collection('obj_weight')
            #print(self.varList_obj)
            self.optimizer_obj = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.graspcnn.loss_obj,
                                                                global_step=global_step, var_list=self.varList_obj)
        with tf.name_scope('optimizer_grasp'):
            self.varList_grasp = tf.get_collection('weight') + tf.get_collection('grasp_weight')
            self.optimizer_grasp = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.graspcnn.loss_grasp,
                                                                global_step=global_step, var_list=self.varList_grasp)
        with tf.name_scope('optimizer_mask'):
            self.varList_mask = tf.get_collection('weight') + tf.get_collection('mask_weight')
            self.optimizer_mask = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.graspcnn.loss_mask,
                                                                global_step=global_step, var_list=self.varList_mask)
        # 3. start tf.session.
        init = tf.global_variables_initializer()
        self.config = tf.ConfigProto() # set config for Session
        self.config.gpu_options.
# general optimization parameters
train_batch_size: 64
val_batch_size: 64

data_split_mode: image_wise # how to split up the data into training vs validation: options are image_wise, stable_pose_wise, object_wise
train_pct: 0.8 # percentage of the data to use for training vs validation
total_pct: 1.0 # percentage of all the files to use
eval_total_train_error: 0

loss: l2
optimizer: rmsprop
base_lr: 0.001
decay_step_multiplier: 0.8   # number of times to go through training datapoints before stepping down decay rate
decay_rate: 0.96
momentum_rate: 0.9

### GQCNN CONFIG ###
graspcnn_config:
  # basic data metrics
  im_height: 50
  im_width: 50
  im_channels: 3
  # needs to match input data mode that was used for training, determines the pose dimensions for the network
  input_data_mode: yuchen_pose_modever()
        self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        logging.info('start session!!!')

    def optimize(self):
        """ Perform optimization """
        start_time = time.time()
        # 1. run setup: dir, training params.
        self._setup()

        # 2. Prepare training Dataset.
        # 2.1. using Dex_net2 dataset.
        #self.num_train = self.loadDataset_obj.prepare_Data(self.experiment_dir)
        #print('optimize: dataset', self.num_train)
        # 2.2 using cornell dataset
        self.num_train = self.loadDataset_obj.prepare_Data()

        # 3. setup tensorflow session/placeholders/queue
        self._setup_tensorflow()

        # 4. optimization
        num_epochs = 200
        num_batches = 300
        self.num_batches_val = 2
        optimize_obj = True
        optimize_grasp = False
        optimize_mask = False
        for i in xrange(num_epochs):
            print('------------------Run epoch:', i, 'in', num_epochs)
            if optimize_obj == True:
                for step_obj in xrange(num_batches):
                    # 1. load minibatch train set.
                    img_train_batch, mask_train_batch, rec_train_batch, obj_train_batch = self.loadDataset_obj.load_Data()
                    #print('optimize: img_train_batch', img_train_batch.shape)
                    #print('optimize: obj_train_batch', obj_train_batch.shape, obj_train_batch[0:3])
                    # 2. optimize: object classfic
                    _, loss_obj, output_obj, learning_rate = self.sess.run([self.optimizer_obj, self.graspcnn.loss_obj,
                                                            self.graspcnn.output_obj, self.learning_rate],
            		                                        feed_dict={self.graspcnn.inputs: img_train_batch,
                                                            self.graspcnn.labels_object: obj_train_batch,
                                                            self.graspcnn.dropout_keep_prob: self.graspcnn._dropout_keep_prob},
            		                                        options=GeneralConstants.timeout_option)
                    #print('optimize: output_obj',output_obj.shape)
                    # 3. tf.summary
                    #logging.info(' object recognition: loss: %.3f, learning rate: %.6f' % (loss_obj, learning_rate))
                    sys.stdout.flush()
                    self.graspcnn.save_scalar(num_batches*i+step_obj, 'train/loss_obj', loss_obj, self.writer)
                    self.graspcnn.save_scalar(num_batches*i+step_obj, 'train/obj_learningRate', learning_rate, self.writer)

                    # 4. evaluate
                    '''
                    if step_obj == num_batches-1:
                        valerror_obj = self.errorrate_obj_valbatch()
                        self.graspcnn.save_scalar(i, 'val/valerror_obj', valerror_obj, self.writer)
                        logging.info(' object recognition: val/valerror_obj: %.3f' % (valerror_obj))
                    '''

            if optimize_grasp == True:
                for step_grasp in xrange(num_batches):
                    # 1. load minibatch train set.
                    img_train_batch, mask_train_batch, rec_train_batch, obj_train_batch = self.loadDataset_obj.load_Data()
                    #print('optimize: img_train_batch', img_train_batch.shape)
                    #print('optimize::rec_train_batch',  rec_train_batch.shape, rec_train_batch[0, :])
                    # 2. optimize: grasp
                    _, loss_grasp, output_grasp, learning_rate = self.sess.run([self.optimizer_grasp, self.graspcnn.loss_grasp,
                                                            self.graspcnn.output_grasp, self.learning_rate],
            		                                        feed_dict={self.graspcnn.inputs: img_train_batch,
                                                            self.graspcnn.labels_grasp: rec_train_batch,
                                                            self.graspcnn.dropout_keep_prob: self.graspcnn._dropout_keep_prob},
            		                                        options=GeneralConstants.timeout_option)
                    #print('optimize: output_grasp', output_grasp.shape, output_grasp[0])
                    # 3. tf.summary
                    #logging.info(' graspPose recognition: loss: %.3f, learning rate: %.6f' % (loss_grasp, learning_rate))
                    sys.stdout.flush()
                    self.graspcnn.save_scalar(num_batches*i+step_grasp, 'train/loss_grasp', loss_grasp, self.writer)
                    self.graspcnn.save_scalar(num_batches*i+step_grasp, 'train/grasp_learningRate', learning_rate, self.writer)
                    # 4. evaluate
                    if step_grasp == num_batches-1:
                        valerror_grasp = self.errorrate_grasp_valbatch()
                        self.graspcnn.save_scalar(i, 'val/valerror_grasp', valerror_grasp, self.writer)
                        logging.info(' object recognition: val/valerror_grasp: %.3f' % (valerror_grasp))

            if optimize_mask == True:
                for step_mask in xrange(num_batches):#for step in training_range: #training_range = 59375
                    # 1. load minibatch train set.
                    img_train_batch, mask_train_batch, rec_train_batch, obj_train_batch = self.loadDataset_obj.load_Data()
                    #print('optimize: img_train_batch', img_train_batch.shape)
                    #print('optimize: mask_train_batch', mask_train_batch.shape, mask_train_batch[0])

                    # 2. optimize: object classfic
                    _, loss_mask, output_mask, learning_rate = self.sess.run([self.optimizer_mask, self.graspcnn.loss_mask,
                                                            self.graspcnn.output_mask, self.learning_rate],
            		                                        feed_dict={self.graspcnn.inputs: img_train_batch,
                                                            self.graspcnn.labels_mask: mask_train_batch,
                                                            self.graspcnn.dropout_keep_prob: self.graspcnn._dropout_keep_prob},
            		                                        options=GeneralConstants.timeout_option)
                    #print('optimize: output_mask', output_mask.shape, output_mask[0])
                    # 3. tf.summary
                    #logging.info(' mask recognition: loss: %.3f, learning rate: %.6f' % (loss_mask, learning_rate))
                    sys.stdout.flush()
                    self.graspcnn.save_scalar(num_batches*i+step_mask, 'train/loss_mask', loss_mask, self.writer)
                    self.graspcnn.save_scalar(num_batches*i+step_mask, 'train/mask_learningRate', learning_rate, self.writer)
                    # 4. evaluate
                    if step_mask == num_batches-1:
                        valerror_mask = self.errorrate_mask_valbatch()
                        self.graspcnn.save_scalar(i, 'val/valerror_mask', valerror_mask, self.writer)
                        logging.info(' object recognition: val/valerror_mask: %.3f' % (valerror_mask))


            # launch tensorboard only after the first iteration
            if not self.tensorboard_has_launched:
            	self.tensorboard_has_launched = True
            	self._launch_tensorboard()
            # save the model at each epoch
        	self.saver.save(self.sess, os.path.join(self.experiment_dir, 'model_%05d.ckpt' %(i)))
        	self.saver.save(self.sess, os.path.join(self.experiment_dir, 'model.ckpt'))

        # update the TrainStatsLogger
        self.saver.save(self.sess, os.path.join(self.experiment_dir, 'model.ckpt'))
        self._close_tensorboard()
        logging.info('Cleaning and Preparing to Exit Optimization')
        self.sess.close()
        del self.saver
        del self.sess
        logging.info('Exiting Optimization')

##############################################################################
    def _setup(self):
        """ Setup for optimization """
        # set up logger
        logging.getLogger().setLevel(logging.INFO)
        # 1.setup output directories
        self._setup_output_dirs()
        # 2.copy config file
        self._copy_config()
        # 3.read  training parameters from config file
        self._read_training_params()

    def errorrate_obj_valbatch(self):
        '''run model in a validation batch'''
        error_rates = []
        for step in xrange(self.num_batches_val):
            # 1. load minibatch train set.
            img_val_batch, mask_val_batch, rec_val_batch, obj_val_batch = self.loadDataset_obj.validation_dataset()

            # 2. prediction with graspNet
            predict_obj= self.sess.run(self.graspcnn.output_obj,feed_dict={self.graspcnn.inputs: img_val_batch,
                                                self.graspcnn.dropout_keep_prob: self.graspcnn._dropout_keep_prob},
                                                options=GeneralConstants.timeout_option)
            # 3. compute error rate.(classfic problem)
            error_rate = 100 -(100*np.sum(predict_obj == obj_val_batch)/predict_obj.shape[0])
            error_rates.append(error_rate)
        print('evaluate:output_obj', predict_obj.shape, predict_obj[0])
        return np.mean(error_rates)

    def errorrate_grasp_valbatch(self):
        '''run model in a validation batch'''
        error_rates = []
        for step in xrange(self.num_batches_val):
            # 1. load minibatch train set.
            img_val_batch, mask_val_batch, rec_val_batch, obj_val_batch = self.loadDataset_obj.validation_dataset()

            # 2. prediction with graspNet
            predict_grasp = self.sess.run(self.graspcnn.output_grasp,feed_dict={self.graspcnn.inputs: img_val_batch,
                                                self.graspcnn.dropout_keep_prob: self.graspcnn._dropout_keep_prob},
                                                options=GeneralConstants.timeout_option)

            # 3. compute error rate.(classfic problem)
            error_rate = 100 -(100*np.sum(predict_obj == obj_val_batch)/predict_obj.shape[0]) #??????????????????????????????????
            error_rates.append(error_rate)
        print('evaluate:predict_grasp', predict_grasp.shape, predict_grasp[0])
        return np.mean(error_rates)

    def errorrate_mask_valbatch(self):
        '''run model in a validation batch'''
        error_rates = []
        for step in xrange(self.num_batches_val):
            # 1. load minibatch train set.
            img_val_batch, mask_val_batch, rec_val_batch, obj_val_batch = self.loadDataset_obj.validation_dataset()

            # 2. prediction with graspNet
            predict_mask = self.sess.run(self.graspcnn.output_mask,feed_dict={self.graspcnn.inputs: img_val_batch,
                                                self.graspcnn.dropout_keep_prob: self.graspcnn._dropout_keep_prob},
                                                options=GeneralConstants.timeout_option)

            # 3. compute error rate.(classfic problem)
            error_rate = 100 -(100*np.sum(predict_obj == obj_val_batch)/predict_obj.shape[0]) #??????????????????????????????????
            error_rates.append(error_rate)
        print('evaluate:predict_mask', predict_mask.shape, predict_mask[0])
        return np.mean(error_rates)


    def _read_training_params(self):
        """ Read training parameters from configuration file """
        self.data_dir = self.cfg['dataset_dir']

        self.train_batch_size = self.cfg['train_batch_size']
        self.val_batch_size = self.cfg['val_batch_size']

        self.base_lr = self.cfg['base_lr']
        self.decay_step_multiplier = self.cfg['decay_step_multiplier']
        self.decay_rate = self.cfg['decay_rate']
        self.momentum_rate = self.cfg['momentum_rate']

    def _create_optimizer(self, loss, batch, var_list, learning_rate):
        """ Create optimizer based on config file
        """
        if self.cfg['optimizer'] == 'momentum':
        	return tf.train.MomentumOptimizer(learning_rate, self.momentum_rate).minimize(loss,  global_step=batch, var_list=var_list)

        elif self.cfg['optimizer'] == 'adam':
        	return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch,var_list=var_list)

        elif self.cfg['optimizer'] == 'rmsprop':
        	return tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=batch,var_list=var_list)
        else:
        	raise ValueError('Optimizer %s not supported' %(self.cfg['optimizer']))

    def _launch_tensorboard(self):
		""" Launches Tensorboard to visualize training """
		logging.info("Launching Tensorboard, Please navigate to localhost:6006 in your favorite web browser to view summaries")
		os.system('tensorboard --logdir=' + self.summary_dir + " &>/dev/null &")

    def _close_tensorboard(self):
        """ Closes Tensorboard """
        logging.info('Closing Tensorboard')
        tensorboard_pid = os.popen('pgrep tensorboard').read()
        os.system('kill ' + tensorboard_pid)

    def _setup_output_dirs(self):
        """ Setup output directories """
        # setup general output directory
        output_dir = self.cfg['output_dir']
        if not os.path.exists(output_dir):
        	os.mkdir(output_dir)
        experiment_id = utils.gen_experiment_id()
        self.experiment_dir = os.path.join(output_dir, 'model_%s' %(experiment_id))
        if not os.path.exists(self.experiment_dir):
        	os.mkdir(self.experiment_dir)
        self.summary_dir = os.path.join(self.experiment_dir, 'tensorboard_summaries')
        if not os.path.exists(self.summary_dir):
        	os.mkdir(self.summary_dir)
        else:
        	# if the summary directory already exists, clean it out by deleting all files in it
        	# we don't want tensorboard to get confused with old logs while debugging with the same directory
        	old_files = os.listdir(self.summary_dir)
        	for file in old_files:
        		os.remove(os.path.join(self.summary_dir, file))

        logging.info('Saving model to %s' %(self.experiment_dir))

        # setup filter directory
        self.filter_dir = os.path.join(self.experiment_dir, 'filters')
        if not os.path.exists(self.filter_dir):
        	os.mkdir(self.filter_dir)

    def _copy_config(self):
        """ Keep a copy of original config files """

        out_config_filename = os.path.join(self.experiment_dir, 'config.json')
        tempOrderedDict = collections.OrderedDict()
        for key in self.cfg.keys():
        	tempOrderedDict[key] = self.cfg[key]
        with open(out_config_filename, 'w') as outfile:
        	json.dump(tempOrderedDict, outfile)
        this_filename = sys.argv[0]
        out_train_filename = os.path.join(self.experiment_dir, 'training_script.py')
        shutil.copyfile(this_filename, out_train_filename)
