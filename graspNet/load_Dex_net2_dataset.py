#!/usr/bin/env python
# coding=utf-8

"""
load the Dex_net2.0 dataset.
Author: yuchen
"""
import argparse
import copy
import cv2
import json
import logging
import numbers
import numpy as np
import cPickle as pkl
import os
import random
import scipy.misc as sm
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as snm
import scipy.stats as ss
import skimage.draw as sd
import signal
import sys
import shutil
import threading
import time
import urllib
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from autolab_core import YamlConfig
import autolab_core.utils as utils
import collections
from graspNet.optimizer_constants_yuchen import ImageMode, TrainingMode, PreprocMode, InputDataMode, GeneralConstants, ImageFileTemplates

import IPython

class loadDataset(object):
    """docstring for loadDataset."""
    def __init__(self, graspcnn, config):
        self.graspcnn = graspcnn
        self.cfg = config

    def prepare_Data(self, experiment_dir):
        """ Setup for dataset """
        self.data_dir = self.cfg['dataset_dir']
        self.image_mode = self.cfg['image_mode']
        self.data_split_mode = self.cfg['data_split_mode']
        self.train_pct = self.cfg['train_pct']
        self.total_pct = self.cfg['total_pct']
        self.decay_step_multiplier = self.cfg['decay_step_multiplier']
        self.train_batch_size = self.cfg['train_batch_size']
        self.max_training_examples_per_load = self.cfg['max_training_examples_per_load']


        self.experiment_dir = experiment_dir
        # 1.setup denoising and synthetic data parameters
        self._setup_denoising_and_synthetic()

        # 2.load image and pose data files (yuchen:fron datafile dir load dataset)
        self._setup_data_filenames()
        # ３．read data parameters from config file
        self._read_data_params()
        # compute train/test indices based on how the data is to be split
        self._compute_indices_image_wise()

        # compute means, std's, and normalization metrics
        self._compute_data_metrics()
        return self.num_train

    def load_Data(self):
        """ Loads a batch of images for training """
        # read parameters of gaussian process
        self.gp_rescale_factor = self.cfg['gaussian_process_scaling_factor']
        self.gp_sample_height = int(self.im_height / self.gp_rescale_factor)
        self.gp_sample_width = int(self.im_width / self.gp_rescale_factor)
        self.gp_num_pix = self.gp_sample_height * self.gp_sample_width
        self.gp_sigma = self.cfg['gaussian_process_sigma']

        # loop through data
        num_queued = 0
        start_i = 0
        end_i = 0
        file_num = 0

        # init buffers
        train_data = np.zeros([self.train_batch_size, self.im_height, self.im_width,
        						self.num_tensor_channels]).astype(np.float32)
        train_poses = np.zeros([self.train_batch_size, self.pose_dim]).astype(np.float32)

        while start_i < self.train_batch_size:#64
        	# 1. compute num remaining
        	num_remaining = self.train_batch_size - num_queued

        	# 2. randomly choice a file from all files
        	file_num = np.random.choice(len(self.im_filenames_copy), size=1)[0]
        	train_data_filename = self.im_filenames_copy[file_num]
        	self.train_data_arr = np.load(os.path.join(self.data_dir, train_data_filename))[
        							 'arr_0'].astype(np.float32)
        	self.train_poses_arr = np.load(os.path.join(self.data_dir, self.pose_filenames_copy[file_num]))[
        							  'arr_0'].astype(np.float32)

        	# 3. get batch indices uniformly at random
        	train_ind = self.train_index_map[train_data_filename]
        	np.random.shuffle(train_ind)
        	upper = min(num_remaining, train_ind.shape[
        				0], self.max_training_examples_per_load)
        	ind = train_ind[:upper]
        	num_loaded = ind.shape[0]
        	end_i = start_i + num_loaded

        	# 4. subsample data
        	self.train_data_arr = self.train_data_arr[ind, ...]
        	self.train_poses_arr = self.train_poses_arr[ind, :]
        	#self.train_label_arr = self.train_label_arr[ind]
        	self.num_images = self.train_data_arr.shape[0]

            # 5. propcess the sampled data
        	# add noises to images
        	self._distort(num_loaded)
        	# subtract mean
        	self.train_data_arr = (self.train_data_arr - self.data_mean) / self.data_std
        	self.train_poses_arr = (self.train_poses_arr - self.pose_mean) / self.pose_std
        	# enqueue training data batch
        	train_data[start_i:end_i, ...] = np.copy(self.train_data_arr)
        	train_poses[start_i:end_i,:] = self._read_pose_data(np.copy(self.train_poses_arr), self.input_data_mode)

            # 6. for next
        	del self.train_data_arr
        	del self.train_poses_arr
        	# update start index
        	start_i = end_i
        	num_queued += num_loaded
        #print('shape of training data and pose:', train_data.shape(), train_poses.shape())
        return train_data, train_poses

#-------------------
    def _setup_denoising_and_synthetic(self):
        """ Setup denoising and synthetic data parameters """
        if self.cfg['multiplicative_denoising']:
        	self.gamma_shape = self.cfg['gamma_shape']
        	self.gamma_scale = 1.0 / self.gamma_shape

    def _setup_data_filenames(self):
        """ Setup image and pose data filenames, subsample files, check validity of filenames/image mode """
        # read in filenames of training data(poses, images, labels)
        logging.info('Reading filenames...')
        all_filenames = os.listdir(self.data_dir)
        if self.image_mode== ImageMode.BINARY:
        	self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.binary_im_tensor_template) > -1]
        elif self.image_mode== ImageMode.DEPTH:
        	self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tensor_template) > -1]
        elif self.image_mode== ImageMode.BINARY_TF:
        	self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.binary_im_tf_tensor_template) > -1]
        elif self.image_mode== ImageMode.COLOR_TF:
        	self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.color_im_tf_tensor_template) > -1]
        elif self.image_mode== ImageMode.GRAY_TF:
        	self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.gray_im_tf_tensor_template) > -1]
        elif self.image_mode== ImageMode.DEPTH_TF:
        	self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tf_tensor_template) > -1]
        elif self.image_mode== ImageMode.DEPTH_TF_TABLE:
        	self.im_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.depth_im_tf_table_tensor_template) > -1]
        else:
        	raise ValueError('Image mode %s not supported.' %(self.image_mode))

        self.pose_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.hand_poses_template) > -1]
        #self.label_filenames = [f for f in all_filenames if f.find(self.target_metric_name) > -1]
        self.obj_id_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.object_labels_template) > -1]
        self.stable_pose_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.pose_labels_template) > -1]

        self.im_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.pose_filenames.sort(key = lambda x: int(x[-9:-4]))
        #self.label_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.obj_id_filenames.sort(key = lambda x: int(x[-9:-4]))
        self.stable_pose_filenames.sort(key = lambda x: int(x[-9:-4]))

        # check valid filenames
        if len(self.im_filenames) == 0 or len(self.pose_filenames) == 0 or len(self.stable_pose_filenames) == 0:
        	raise ValueError('One or more required training files in the dataset could not be found.')
        if len(self.obj_id_filenames) == 0:
        	self.obj_id_filenames = None

        # subsample files
        self.num_files = len(self.im_filenames)
        num_files_used = int(self.total_pct * self.num_files) #choice 100% from all files
        filename_indices = np.random.choice(self.num_files, size=num_files_used, replace=False)
        filename_indices.sort()
        self.im_filenames = [self.im_filenames[k] for k in filename_indices]
        self.pose_filenames = [self.pose_filenames[k] for k in filename_indices]
        #self.label_filenames = [self.label_filenames[k] for k in filename_indices]
        if self.obj_id_filenames is not None:
        	self.obj_id_filenames = [self.obj_id_filenames[k] for k in filename_indices]
        self.stable_pose_filenames = [self.stable_pose_filenames[k] for k in filename_indices]

        # create copy of image, pose, and label filenames because original cannot be accessed by load and enqueue op in the case that the error_rate_in_batches method is sorting the original
        self.im_filenames_copy = self.im_filenames[:]
        self.pose_filenames_copy = self.pose_filenames[:]
        #self.label_filenames_copy = self.label_filenames[:]
        logging.info('success load data file!')

    def _read_data_params(self):
        """ Read data parameters from configuration file """
        self.train_im_data = np.load(os.path.join(self.data_dir, self.im_filenames[0]))['arr_0']
        self.pose_data = np.load(os.path.join(self.data_dir, self.pose_filenames[0]))['arr_0']
        #self.metric_data = np.load(os.path.join(self.data_dir, self.label_filenames[0]))['arr_0']
        self.images_per_file = self.train_im_data.shape[0]
        self.im_height = self.train_im_data.shape[1]
        self.im_width = self.train_im_data.shape[2]
        self.im_channels = self.train_im_data.shape[3]
        self.im_center = np.array([float(self.im_height-1)/2, float(self.im_width-1)/2])
        self.num_tensor_channels = self.cfg['num_tensor_channels']
        self.pose_shape = self.pose_data.shape[1]

        self.input_data_mode = self.cfg['input_data_mode']
        if self.input_data_mode == InputDataMode.TF_IMAGE:
        	self.pose_dim = 1 # depth
        elif self.input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
        	self.pose_dim = 3 # depth, cx, cy
        elif self.input_data_mode == InputDataMode.RAW_IMAGE:
        	self.pose_dim = 4 # u, v, theta, depth
        elif self.input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
        	self.pose_dim = 6 # u, v, theta, depth cx, cy
        elif self.input_data_mode == InputDataMode.YUCHEN_POSE_MODE:
        	self.pose_dim = 4 # 4 dimensions
        else:
        	raise ValueError('Input data mode %s not understood' %(self.input_data_mode))

        self.num_files = len(self.im_filenames)
        self.num_random_files = min(self.num_files, self.cfg['num_random_files'])
        self.num_categories = 2

    def _compute_indices_image_wise(self):
        """ Compute train and validation indices based on an image-wise split"""
        # get total number of training datapoints
        #print('_compute_indices_image_wise:', self.images_per_file, self.num_files)
        num_datapoints = self.images_per_file * self.num_files # 1000*190
        self.num_train = int(self.train_pct * num_datapoints) #use 80% for training

        # get training and validation indices #yuchen: using indices split files into training and testing files
        all_indices = np.arange(num_datapoints)
        np.random.shuffle(all_indices)
        train_indices = np.sort(all_indices[:self.num_train])
        val_indices = np.sort(all_indices[self.num_train:])

        # make a map of the train and test indices for each file
        logging.info('Computing indices image-wise')
        train_index_map_filename = os.path.join(self.experiment_dir, 'train_indices_image_wise.pkl') #for save dataset
        self.val_index_map_filename = os.path.join(self.experiment_dir, 'val_indices_image_wise.pkl')
        if os.path.exists(train_index_map_filename):
        	self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
        	self.val_index_map = pkl.load(open(self.val_index_map_filename, 'r'))
        else:
        	self.train_index_map = {}
        	self.val_index_map = {}
        	for i, im_filename in enumerate(self.im_filenames):
        		lower = i * self.images_per_file
        		upper = (i+1) * self.images_per_file
        		im_arr = np.load(os.path.join(self.data_dir, im_filename))['arr_0'] #dir/depth_ims_tf_table_00182.npz
        		#obtain: train_index_map and val_index_map
        		self.train_index_map[im_filename] = train_indices[(train_indices >= lower) & (train_indices < upper) &
        												(train_indices - lower < im_arr.shape[0])] - lower
        		self.val_index_map[im_filename] = val_indices[(val_indices >= lower) & (val_indices < upper) &
        												(val_indices - lower < im_arr.shape[0])] - lower
        	pkl.dump(self.train_index_map, open(train_index_map_filename, 'w')) #save to file.
        	pkl.dump(self.val_index_map, open(self.val_index_map_filename, 'w'))

    def _compute_data_metrics(self):
        """ Calculate image mean, image std, pose mean, pose std, normalization params """
        # compute data mean
        logging.info('Computing image mean')
        mean_filename = os.path.join(self.experiment_dir, 'mean.npy')
        std_filename = os.path.join(self.experiment_dir, 'std.npy')
        if self.cfg['fine_tune']:
        	self.data_mean = self.graspcnn.get_im_mean()
        	self.data_std = self.graspcnn.get_im_std()
        else:
        	self.data_mean = 0
        	self.data_std = 0
        	random_file_indices = np.random.choice(self.num_files, size=self.num_random_files, replace=False)
        	num_summed = 0
        	for k in random_file_indices.tolist():
        		im_filename = self.im_filenames[k]
        		im_data = np.load(os.path.join(self.data_dir, im_filename))['arr_0']
        		self.data_mean += np.sum(im_data[self.train_index_map[im_filename], :, :, :])
        		num_summed += im_data[self.train_index_map[im_filename], :, :, :].shape[0]
        	self.data_mean = self.data_mean / (num_summed * self.im_height * self.im_width)
        	np.save(mean_filename, self.data_mean)

        	for k in random_file_indices.tolist():
        		im_filename = self.im_filenames[k]
        		im_data = np.load(os.path.join(self.data_dir, im_filename))['arr_0']
        		self.data_std += np.sum((im_data[self.train_index_map[im_filename], :, :, :] - self.data_mean)**2)
        	self.data_std = np.sqrt(self.data_std / (num_summed * self.im_height * self.im_width))
        	np.save(std_filename, self.data_std)

        # compute pose mean
        logging.info('Computing pose mean')
        self.pose_mean_filename = os.path.join(self.experiment_dir, 'pose_mean.npy')
        self.pose_std_filename = os.path.join(self.experiment_dir, 'pose_std.npy')
        if self.cfg['fine_tune']:
        	self.pose_mean = self.graspcnn.get_pose_mean()
        	self.pose_std = self.graspcnn.get_pose_std()
        else:
        	self.pose_mean = np.zeros(self.pose_shape)
        	self.pose_std = np.zeros(self.pose_shape)
        	num_summed = 0
        	random_file_indices = np.random.choice(self.num_files, size=self.num_random_files, replace=False)
        	for k in random_file_indices.tolist():
        		im_filename = self.im_filenames[k]
        		pose_filename = self.pose_filenames[k]
        		self.pose_data = np.load(os.path.join(self.data_dir, pose_filename))['arr_0']
        		self.pose_mean += np.sum(self.pose_data[self.train_index_map[im_filename],:], axis=0)
        		num_summed += self.pose_data[self.train_index_map[im_filename]].shape[0]
        	self.pose_mean = self.pose_mean / num_summed

        	for k in random_file_indices.tolist():
        		im_filename = self.im_filenames[k]
        		pose_filename = self.pose_filenames[k]
        		self.pose_data = np.load(os.path.join(self.data_dir, pose_filename))['arr_0']
        		self.pose_std += np.sum((self.pose_data[self.train_index_map[im_filename],:] - self.pose_mean)**2, axis=0)
        	self.pose_std = np.sqrt(self.pose_std / num_summed)
        	self.pose_std[self.pose_std==0] = 1.0
        	np.save(self.pose_mean_filename, self.pose_mean)
        	np.save(self.pose_std_filename, self.pose_std)

        if self.cfg['fine_tune']:
        	out_mean_filename = os.path.join(self.experiment_dir, 'mean.npy')
        	out_std_filename = os.path.join(self.experiment_dir, 'std.npy')
        	out_pose_mean_filename = os.path.join(self.experiment_dir, 'pose_mean.npy')
        	out_pose_std_filename = os.path.join(self.experiment_dir, 'pose_std.npy')
        	np.save(out_mean_filename, self.data_mean)
        	np.save(out_std_filename, self.data_std)
        	np.save(out_pose_mean_filename, self.pose_mean)
        	np.save(out_pose_std_filename, self.pose_std)
        ''''
        # update graspcnn im mean & std
        #self.graspcnn.update_im_mean(self.data_mean)
        #self.graspcnn.update_im_std(self.data_std)

        # update graspcnn pose_mean and pose_std according to data_mode
        if self.input_data_mode == InputDataMode.TF_IMAGE:
        	# depth
        	if isinstance(self.pose_mean, numbers.Number) or self.pose_mean.shape[0] == 1:
        		self.graspcnn.update_pose_mean(self.pose_mean)
        		self.graspcnn.update_pose_std(self.pose_std)
        	else:
        		self.graspcnn.update_pose_mean(self.pose_mean[2])
        		self.graspcnn.update_pose_std(self.pose_std[2])
        elif self.input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
        	# depth, cx, cy
        	self.graspcnn.update_pose_mean(np.concatenate([self.pose_mean[2:3], self.pose_mean[4:6]]))
        	self.graspcnn.update_pose_std(np.concatenate([self.pose_std[2:3], self.pose_std[4:6]]))
        elif self.input_data_mode == InputDataMode.RAW_IMAGE:
        	# u, v, depth, theta
        	self.graspcnn.update_pose_mean(self.pose_mean[:4])
        	self.graspcnn.update_pose_std(self.pose_std[:4])
        elif self.input_data_mode == InputDataMode.RAW_IMAGE_PERSPECTIVE:
        	# u, v, depth, theta, cx, cy
        	self.graspcnn.update_pose_mean(self.pose_mean[:6])
        	self.graspcnn.update_pose_std(self.pose_std[:6])
        elif self.input_data_mode == InputDataMode.YUCHEN_POSE_MODE:# yuchen
        	# 4 dimensions.
        	self.graspcnn.update_pose_mean(self.pose_mean[:4])
        	self.graspcnn.update_pose_std(self.pose_std[:4])
        '''

    def _distort(self, num_loaded):
        """ Adds noise to a batch of images """
        # denoising and synthetic data generation
        if self.cfg['multiplicative_denoising']:
        	mult_samples = ss.gamma.rvs(self.gamma_shape, scale=self.gamma_scale, size=num_loaded)
        	mult_samples = mult_samples[:,np.newaxis,np.newaxis,np.newaxis]
        	self.train_data_arr = self.train_data_arr * np.tile(mult_samples,
        									[1, self.im_height, self.im_width, self.im_channels])

        # randomly dropout regions of the image for robustness
        if self.cfg['image_dropout']:
        	for i in range(self.num_images):
        		if np.random.rand() < self.cfg['image_dropout_rate']:
        			train_image = self.train_data_arr[i,:,:,0]
        			nonzero_px = np.where(train_image > 0)
        			nonzero_px = np.c_[nonzero_px[0], nonzero_px[1]]
        			num_nonzero = nonzero_px.shape[0]
        			num_dropout_regions = ss.poisson.rvs(self.cfg['dropout_poisson_mean'])

        			# sample ellipses
        			dropout_centers = np.random.choice(num_nonzero, size=num_dropout_regions)
        			x_radii = ss.gamma.rvs(self.cfg['dropout_radius_shape'], scale=self.cfg['dropout_radius_scale'], size=num_dropout_regions)
        			y_radii = ss.gamma.rvs(self.cfg['dropout_radius_shape'], scale=self.cfg['dropout_radius_scale'], size=num_dropout_regions)

        			# set interior pixels to zero
        			for j in range(num_dropout_regions):
        				ind = dropout_centers[j]
        				dropout_center = nonzero_px[ind, :]
        				x_radius = x_radii[j]
        				y_radius = y_radii[j]
        				dropout_px_y, dropout_px_x = sd.ellipse(dropout_center[0], dropout_center[1], y_radius, x_radius, shape=train_image.shape)
        				train_image[dropout_px_y, dropout_px_x] = 0.0
        			self.train_data_arr[i,:,:,0] = train_image

        # dropout a region around the areas of the image with high gradient
        if self.cfg['gradient_dropout']:
        	for i in range(self.num_images):
        		if np.random.rand() < self.cfg['gradient_dropout_rate']:
        			train_image = self.train_data_arr[i,:,:,0]
        			grad_mag = sf.gaussian_gradient_magnitude(train_image, sigma=self.cfg['gradient_dropout_sigma'])
        			thresh = ss.gamma.rvs(self.cfg['gradient_dropout_shape'], self.cfg['gradient_dropout_scale'], size=1)
        			high_gradient_px = np.where(grad_mag > thresh)
        			train_image[high_gradient_px[0], high_gradient_px[1]] = 0.0
        		self.train_data_arr[i,:,:,0] = train_image

        # add correlated Gaussian noise
        if self.cfg['gaussian_process_denoising']:
        	for i in range(self.num_images):
        		if np.random.rand() < self.cfg['gaussian_process_rate']:
        			train_image = self.train_data_arr[i,:,:,0]
        			gp_noise = ss.norm.rvs(scale=self.gp_sigma, size=self.gp_num_pix).reshape(self.gp_sample_height,
        									self.gp_sample_width)
        			gp_noise = sm.imresize(gp_noise, self.gp_rescale_factor, interp='bicubic', mode='F')
        			train_image[train_image > 0] += gp_noise[train_image > 0]
        			self.train_data_arr[i,:,:,0] = train_image

        # run open and close filters to
        if self.cfg['morphological']:
        	for i in range(self.num_images):
        		train_image = self.train_data_arr[i,:,:,0]
        		sample = np.random.rand()
        		morph_filter_dim = ss.poisson.rvs(self.cfg['morph_poisson_mean'])
        		if sample < self.cfg['morph_open_rate']:
        			train_image = snm.grey_opening(train_image, size=morph_filter_dim)
        		else:
        			closed_train_image = snm.grey_closing(train_image, size=morph_filter_dim)

        			# set new closed pixels to the minimum depth, mimicing the table
        			new_nonzero_px = np.where((train_image == 0) & (closed_train_image > 0))
        			closed_train_image[new_nonzero_px[0], new_nonzero_px[1]] = np.min(train_image[train_image>0])
        			train_image = closed_train_image.copy()

        		self.train_data_arr[i,:,:,0] = train_image

        # randomly dropout borders of the image for robustness
        if self.cfg['border_distortion']:
        	for i in range(self.num_images):
        		train_image = self.train_data_arr[i,:,:,0]
        		grad_mag = sf.gaussian_gradient_magnitude(train_image, sigma=self.cfg['border_grad_sigma'])
        		high_gradient_px = np.where(grad_mag > self.cfg['border_grad_thresh'])
        		high_gradient_px = np.c_[high_gradient_px[0], high_gradient_px[1]]
        		num_nonzero = high_gradient_px.shape[0]
        		num_dropout_regions = ss.poisson.rvs(self.cfg['border_poisson_mean'])

        		# sample ellipses
        		dropout_centers = np.random.choice(num_nonzero, size=num_dropout_regions)
        		x_radii = ss.gamma.rvs(self.cfg['border_radius_shape'], scale=self.cfg['border_radius_scale'], size=num_dropout_regions)
        		y_radii = ss.gamma.rvs(self.cfg['border_radius_shape'], scale=self.cfg['border_radius_scale'], size=num_dropout_regions)

        		# set interior pixels to zero or one
        		for j in range(num_dropout_regions):
        			ind = dropout_centers[j]
        			dropout_center = high_gradient_px[ind, :]
        			x_radius = x_radii[j]
        			y_radius = y_radii[j]
        			dropout_px_y, dropout_px_x = sd.ellipse(dropout_center[0], dropout_center[1], y_radius, x_radius, shape=train_image.shape)
        			if np.random.rand() < 0.5:
        				train_image[dropout_px_y, dropout_px_x] = 0.0
        			else:
        				train_image[dropout_px_y, dropout_px_x] = train_image[dropout_center[0], dropout_center[1]]

        		self.train_data_arr[i,:,:,0] = train_image

        # randomly replace background pixels with constant depth
        if self.cfg['background_denoising']:
        	for i in range(self.num_images):
        		train_image = self.train_data_arr[i,:,:,0]
        		if np.random.rand() < self.cfg['background_rate']:
        			train_image[train_image > 0] = self.cfg['background_min_depth'] + (self.cfg['background_max_depth'] - self.cfg['background_min_depth']) * np.random.rand()

        # symmetrize images
        if self.cfg['symmetrize']:
        	for i in range(self.num_images):
        		train_image = self.train_data_arr[i,:,:,0]
        		# rotate with 50% probability
        		if np.random.rand() < 0.5:
        			theta = 180.0
        			rot_map = cv2.getRotationMatrix2D(tuple(self.im_center), theta, 1)
        			train_image = cv2.warpAffine(train_image, rot_map, (self.im_height, self.im_width), flags=cv2.INTER_NEAREST)
        			'''
        			if self.pose_dim > 1:
        				self.train_poses_arr[i,4] = -self.train_poses_arr[i,4]
        				self.train_poses_arr[i,5] = -self.train_poses_arr[i,5]
        			'''
        		# reflect left right with 50% probability
        		if np.random.rand() < 0.5:
        			train_image = np.fliplr(train_image)
        			'''
        			if self.pose_dim > 1:
        				self.train_poses_arr[i,5] = -self.train_poses_arr[i,5]
        			'''
        		# reflect up down with 50% probability
        		if np.random.rand() < 0.5:
        			train_image = np.flipud(train_image)
        			'''
        			if self.pose_dim > 1:
        				self.train_poses_arr[i,4] = -self.train_poses_arr[i,4]
        			'''
        		self.train_data_arr[i,:,:,0] = train_image
        return self.train_data_arr, self.train_poses_arr

    def _read_pose_data(self, pose_arr, input_data_mode):
        """ Read the pose data and slice it according to the specified input_data_mode
        """
        if input_data_mode == InputDataMode.TF_IMAGE:
        	return pose_arr[:,2:3]
        elif input_data_mode == InputDataMode.TF_IMAGE_PERSPECTIVE:
        	return np.c_[pose_arr[:,2:3], pose_arr[:,4:6]]
        elif input_data_mode == InputDataMode.YUCHEN_POSE_MODE:
        	return pose_arr
        else:
        	raise ValueError('Input data mode %s not supported. The RAW_* input data modes have been deprecated.' %(input_data_mode))
