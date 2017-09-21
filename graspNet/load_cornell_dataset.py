#!/usr/bin/env python
# coding=utf-8
import numpy as np
from scipy import misc
from PIL import Image, ImageDraw, ImageFont
import shutil
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd

def data_path():
    return '/informatik2/tams/home/deng/catkin_ws/src/Robot_grasp/grasp_dataset/'

class loadDataset(object):
    """docstring for ."""
    def __init__(self, train_config):
        self.data_path = '/informatik2/tams/home/deng/catkin_ws/src/Robot_grasp/'
        self.config = train_config
        #self.num_train = 400
        #self.num_validation = 50
        self.scale = 6
        self.train_batch_size = self.config['train_batch_size']
        self.val_batch_size = self.config['val_batch_size']

        self.min_dir_n = 1
        self.max_dir_n = 5
        self.max_pic_n = 99

        self.im_height = self.config['graspcnn_config']['im_height']
        self.im_width = self.config['graspcnn_config']['im_width']
        self.im_channels = self.config['graspcnn_config']['im_channels']

        self.grasp_dim = self.config['graspcnn_config']['dim_grasp']

        self.num_objClasses = 34

    # prepare dataset
    def prepare_dataset(self, min_directory_n, max_directory_n, max_picture_n):
        img_dataset = []
        rec_dataset = []
        obj_dataset = []
        mask_dataset = []

        for i in range(min_directory_n, max_directory_n+1): # 1. file name
            if i == 9 or i == 19 or i == 29:
                max_picture_n = 49
            elif i == 10 or i == 20 or i == 30:
                max_picture_n = 34
            else:
                max_picture_n = 99
            print "loading file from directory No: " + '0'+ str(i) + ', ' +str(max_picture_n)
            for j in range(max_picture_n+1): # 2. picture num
                data_label = self.label_handling(i,j)
                img_list = self.load_picture(data_label)
                mask_list = self.load_mask(data_label)
                rec_array = self.load_rectangle(data_label,neg_or_pos = 1) # only use the positive grasp
                obj_labels = self.load_object(data_label)

                for n in range(len(rec_array)): # 3. diffrent positive grasp
                    rec_list = rec_array[n].tolist()
                    img_dataset.append(img_list)
                    mask_dataset.append(mask_list)
                    rec_dataset.append(rec_list)
                    obj_dataset.append(obj_labels[j])

        img_dataset = np.array(img_dataset)
        mask_dataset = np.array(mask_dataset)
        rec_dataset = np.array(rec_dataset)
        obj_dataset = np.array(obj_dataset)
        return img_dataset, mask_dataset, rec_dataset, obj_dataset

    # generate train and validation dataset
    def prepare_Data(self):
        img_train = []
        mask_train = []
        rec_train = []
        obj_train = []
        img_validation = []
        mask_validation = []
        rec_validation = []
        obj_validation = []

        img_dataset, mask_dataset, rec_dataset, obj_dataset = self.prepare_dataset(self.min_dir_n, self.max_dir_n,self.max_pic_n)

        print " "
        print "loaded learning dataset:"
        print "directory: " + str(self.min_dir_n) +"-"+ str(self.max_dir_n) + " picture: 0-" + str(self.max_pic_n)
        print "total data amount: " + str(len(img_dataset))

        indexes = np.random.permutation(len(img_dataset))
        self.num_train = int(0.8 * len(img_dataset))
        self.num_validation = int(0.2 * len(img_dataset))

        for i in range(self.num_train + self.num_validation):
            if i < self.num_train:
                img_train.append(img_dataset[indexes[i]])
                mask_train.append(mask_dataset[indexes[i]])
                rec_train.append(rec_dataset[indexes[i]])
                obj_train.append(obj_dataset[indexes[i]])
            else:
                img_validation.append(img_dataset[indexes[i]])
                mask_validation.append(mask_dataset[indexes[i]])
                rec_validation.append(rec_dataset[indexes[i]])
                obj_validation.append(obj_dataset[indexes[i]])

        self.img_train = np.array(img_train, dtype = np.float32)
        self.mask_train = np.array(mask_train)
        self.rec_train = np.asarray(rec_train)
        self.obj_train = np.array(obj_train)

        self.img_validation = np.array(img_validation, dtype = np.float32)
        self.mask_validation = np.array(mask_validation)
        self.rec_validation = np.asarray(rec_validation)
        self.obj_validation = np.array(obj_validation)

        print "train_N: " + str(len(img_train))
        print "validation_N: " + str(len(img_validation))
        # don't need to use: self.Y_train,self.Y_validation
        #print('prepare_Data: self.obj_train', self.obj_train[0:5])
        return self.num_train

    def load_Data(self):
        # 1. randomly choise form a batch
        img_train_batch = []
        mask_train_batch = []
        rec_train_batch = []
        obj_train_batch = []
        for i in xrange(self.train_batch_size):
            random_num = np.random.choice(self.num_train)

            img_train_batch.append(self.img_train[random_num])
            mask_train_batch.append(self.mask_train[random_num])
            rec_train_batch.append(self.rec_train[random_num])
            obj_train_batch.append(self.obj_train[random_num])
        #print('load_Data:rec_train_batch',rec_train_batch.shape)
        #print('load_Data:img_train_batch',img_train_batch.shape)

        self.rec_train_batch = np.asarray(rec_train_batch).reshape(self.train_batch_size,self.grasp_dim).astype(np.float32)
        img_train_batch = np.asarray(img_train_batch).reshape(self.train_batch_size,self.im_channels,self.im_height,self.im_width).astype(np.float32)
        self.img_train_batch = np.transpose(img_train_batch, (0,2,3,1))
        self.mask_train_batch = np.asarray(mask_train_batch).reshape(self.train_batch_size, self.im_height*self.im_width).astype(np.int16)
        self.obj_train_batch = np.asarray(obj_train_batch).reshape(self.train_batch_size, self.num_objClasses).astype(np.int16)
        #print('load_Data:img_train_batch', self.obj_train_batch.shape)
        return self.img_train_batch, self.mask_train_batch, self.rec_train_batch, self.obj_train_batch

    # generate vaidation dataset
    def validation_dataset(self):
        # 1. randomly choise form a batch
        img_val_batch = []
        mask_val_batch = []
        rec_val_batch = []
        obj_val_batch = []
        for i in xrange(self.val_batch_size):
            random_num = np.random.choice(self.num_validation)
            img_val_batch.append(self.img_validation[random_num])
            mask_val_batch.append(self.mask_validation[random_num])
            rec_val_batch.append(self.rec_validation[random_num])
            obj_val_batch.append(self.obj_validation[random_num])

        self.rec_val_batch = np.asarray(rec_val_batch).reshape(self.val_batch_size,self.grasp_dim).astype(np.float32)
        img_val_batch = np.asarray(img_val_batch).reshape(self.val_batch_size,self.im_channels,self.im_height,self.im_width).astype(np.float32)
        self.img_val_batch = np.transpose(img_val_batch, (0,2,3,1))
        self.mask_val_batch = np.asarray(mask_val_batch).reshape(self.val_batch_size, self.im_height*self.im_width).astype(np.int16)
        self.obj_val_batch = np.asarray(obj_val_batch).reshape(self.val_batch_size, self.num_objClasses).astype(np.int16)
        return self.img_val_batch, self.mask_val_batch, self.rec_val_batch, self.obj_val_batch


    # label preparation:
    def label_handling(self, data_label_1,data_label_2):
        '''data_label_1: file name, data_label_2: picture number '''
        data_label = []

        if data_label_1 < 10 :
            data_label.append(str(0)+str(data_label_1))
        else:
            data_label.append(str(data_label_1))

        if data_label_2 < 10 :
            data_label.append(str(0)+str(data_label_2))
        else:
            data_label.append(str(data_label_2))

        return data_label

    # load picture data
    def load_picture(self, data_label):
        #load image
        path = data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'crop.png'
        img =Image.open(path)
        resize_img = img.resize((img.size[0]/self.scale,img.size[1]/self.scale))
        #resize_img = img.resize((self.im_height,self.im_width))
        #img.show()
        #resize_img.show()
        #print('load_picture:', 'origin:', img.size, 'new:', resize_img.size)

        img_array = np.asanyarray(resize_img, dtype=np.float32)
        img_shape = img_array.shape
        img_array = np.reshape(img_array,(img_shape[2]*img_shape[1]*img_shape[0],1)) # 3*100*100
        img_list = []
        for i in range(len(img_array)):
            img_list.append(img_array[i][0]/255.0)
        return img_list

    # load mask
    def load_mask(self, data_label):
        path = data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'mcrop.png'
        img =Image.open(path)
        resize_img = img.resize((img.size[0]/self.scale,img.size[1]/self.scale))
        #img.show()
        #resize_img.show()
        #print('load_picture:', 'origin:', img.size, 'new:', resize_img.size)

        img_array = np.asanyarray(resize_img,dtype=np.float32)
        img_shape = img_array.shape
        img_array = np.reshape(img_array,(img_shape[1]*img_shape[0],1)) # 100*100
        img_array = img_array/255.0
        return img_array

    # load rectangle data
    def load_rectangle(self, data_label,neg_or_pos):
        neg_pos = ['neg','pos']
        rec_list = []
        path = data_path()+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'c'+neg_pos[neg_or_pos]+'.txt'
        for line in open(path).readlines():
            rec_str = line.split(' ')
            x = float(rec_str[0]) #crop operation.
            y = float(rec_str[1])
            rec_list.append([x/self.scale, y/self.scale]) # zoom operation.
        rec_array = np.array(rec_list, dtype=np.float32).reshape(len(rec_list)/4, 8) #each rectangle represented by 4 points.

        # 2. four coordinate change to grasp pose
        rec_pose = []
        for i in range(rec_array.shape[0]):
            tmp_rec = np.copy(rec_array[i])
            # point coordinate ----> center, angle , width , height
            # point 1= [0,1], point 2 =[2,3], point 3 = [4,5], point 4 = [6, 7]
            point14_x = (tmp_rec[0] + tmp_rec[6])/2
            point14_y = (tmp_rec[1] + tmp_rec[7])/2
            point23_x = (tmp_rec[2] + tmp_rec[4])/2
            point23_y = (tmp_rec[3] + tmp_rec[5])/2
            center_x =  (point14_x + point23_x)/2
            center_y = (point14_y + point23_y)/2
            width = np.sqrt(np.square(tmp_rec[0] - tmp_rec[2]) + np.square(tmp_rec[1] - tmp_rec[3])) #point1 - point2
            height = np.sqrt(np.square(tmp_rec[0] - tmp_rec[6]) + np.square(tmp_rec[1] - tmp_rec[7])) # point1 - point4
            theta = np.arctan2((point23_y-point14_y), (point23_x-point14_x)) #*180 / np.pi
            rec_pose.append([center_x, center_y, theta, width, height])
        rec_pose_array = np.array(rec_pose, dtype=np.float32)
        return rec_pose_array

    def load_object(self, data_label):
        #load object label
        path = data_path()+data_label[0]+'/pcd'+data_label[0] + '.csv'
        data_pd = pd.read_csv(path)
        #print('load_object: data_pd', data_pd.head())
        labels_flat = data_pd.values[:,1]

        labels = self.dense_to_one_hot(labels_flat, self.num_objClasses)
        labels = labels.astype(np.uint8)
        return labels


    def dense_to_one_hot(self, labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes #1500
        #print('dense_to_one_hot: size', 'num_labels', num_labels, 'num_classes', num_classes)
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def display_all(self):
        dataset_path = '/informatik2/tams/home/deng/catkin_ws/src/Robot_grasp/'
        file_num = 1
        img_num = 5
        data_label = self.label_handling(file_num, img_num)

        #---------------image and rectangle
        _, img = self.load_picture(data_label)

        # -------------load rectangle for img
        neg_or_pos = 1
        rec_array = self.load_rectangle(data_label, neg_or_pos)
        rec = rec_array[0].tolist()
        xc = rec[0] #[center_x, center_y, theta, width, height]  -->coordinate
        yc = rec[1]
        theta = rec[2]
        a = rec[3]
        b = rec[4]
        rec_point = [0,0,0,0,0,0,0,0]
        rec_point[0] = a*np.cos(theta)-b*np.sin(theta)+xc
        rec_point[1] = a*np.sin(theta)+b*np.cos(theta)+yc
        rec_point[2] = -a*np.cos(theta)-b*np.sin(theta)+xc
        rec_point[3] = -a*np.sin(theta)+b*np.cos(theta)+yc
        rec_point[4] = -a*np.cos(theta)+b*np.sin(theta)+xc
        rec_point[5] = -a*np.sin(theta)-b*np.cos(theta)+yc
        rec_point[6] = a*np.cos(theta)+b*np.sin(theta)+xc
        rec_point[7] = a*np.sin(theta)-b*np.cos(theta)+yc
        draw_rec = ImageDraw.Draw(img)
        draw_rec.line((rec_point[0],rec_point[1])+(rec_point[2],rec_point[3]), fill='yellow', width=2)
        draw_rec.line((rec_point[2],rec_point[3])+(rec_point[4],rec_point[5]), fill='green', width=2)
        draw_rec.line((rec_point[4],rec_point[5])+(rec_point[6],rec_point[7]), fill='red', width=2)
        draw_rec.line((rec_point[6],rec_point[7])+(rec_point[0],rec_point[1]), fill='blue', width=2)
        img.show()

        #------------------mask
        mask_array = self.load_mask(data_label)
        mask_array = mask_array*255
        mask = Image.fromarray(mask_array)
        mask.show()
#main
if __name__ == '__main__':
    from autolab_core import YamlConfig
    import logging
    import argparse
    from visualizer_cornell_yuchen import draw_rec_yuchen

    # setup logger
    logging.getLogger().setLevel(logging.INFO)
    # parse args
    parser = argparse.ArgumentParser(description='Train a grasp Network with TensorFlow')
    parser.add_argument('--config_filename', type=str, default='/informatik2/tams/home/deng/catkin_ws/src/Robot_grasp/graspPlanning_yuchen/cfg/tools/training_yuchen.yaml', help='path to the configuration file')
    args = parser.parse_args()
    config_filename = args.config_filename

    # open config file
    train_config = YamlConfig(config_filename)
    loadDataset_obj = loadDataset(train_config)

    data_label = ['01', '00']
    img_list = loadDataset_obj.display_all()
