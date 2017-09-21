"""
Script with examples for:
1) Training graspNet using Stochastic Gradient Descent
2) Predicting probability of grasp success from images in batches using a pre-trained graspNet model
3) Fine-tuning a graspNet model
4) Analyzing a graspNet model

"""
import argparse
import logging
import time
import os

from autolab_core import YamlConfig
#from gqcnn import GQCNN, SGDOptimizer, GQCNNAnalyzer
from graspNet.neural_networks2_yuchen import graspCNN
from graspNet.sgd_optimizer3_yuchen import SGDOptimizer
#from graspNet.graspcnn_analyzer_yuchen import graspCNNAnalyzer
#from graspNet.load_Dex_net2_dataset import loadDataset
from graspNet.load_cornell_dataset import loadDataset


if __name__ == '__main__':
	# setup logger
	logging.getLogger().setLevel(logging.INFO)
	# parse args
	parser = argparse.ArgumentParser(description='Train a grasp Network with TensorFlow')
	parser.add_argument('--config_filename', type=str, default='cfg/tools/training_yuchen.yaml', help='path to the configuration file')
	args = parser.parse_args()
	config_filename = args.config_filename

    # open config file
	train_config = YamlConfig(config_filename)
	graspcnn_config = train_config['graspcnn_config']

	def get_elapsed_time(time_in_seconds):
		""" Helper function to get elapsed time """
		if time_in_seconds < 60:
			return '%.1f seconds' % (time_in_seconds)
		elif time_in_seconds < 3600:
			return '%.1f minutes' % (time_in_seconds / 60)
		else:
			return '%.1f hours' % (time_in_seconds / 3600)

	# Fine-Tuning

	start_time = time.time()
	model_dir = train_config['model_dir']
	gqcnn = GQCNN.load(model_dir)
	sgdOptimizer = SGDOptimizer(gqcnn, train_config)
	with gqcnn._graph.as_default():
	        sgdOptimizer.optimize()
	logging.info('Total Fine Tuning Time:' + str(get_elapsed_time(time.time() - start_time)))
