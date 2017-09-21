# coding=utf-8
#!/usr/bin/env python
# python library
import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw, ImageFont
# OpenCV
import cv2


def augment_multiple():
    dataset_path = '/informatik2/tams/home/deng/catkin_ws/src/Robot_grasp/grasp_dataset/'
    rotate_angle = -90 #degree

    file_num = 10
    if file_num == 9:
        max_picture_n = 49
    elif file_num == 10:
        max_picture_n = 34
    else:
        max_picture_n = 99
    #max_picture_n =0
    for img_num in range(max_picture_n+1): # 2. picture num
        print(img_num)
        data_label, data_label_aug = label_handling(file_num, img_num)
        #print(data_label, data_label_aug)

        # 1.load image
        path_croped = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'crop.png'
        img_croped = Image.open(path_croped)
        path_rotated = dataset_path+data_label_aug[0]+'/pcd'+data_label_aug[0]+data_label_aug[1]+'crop.png'
        #print('path_rotated', path_rotated)
        img_rotated = img_croped.rotate(rotate_angle)
        img_rotated.save(path_rotated)

        path_maskcroped = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'mcrop.png'
        mask_croped = Image.open(path_maskcroped)
        path_maskrotated = dataset_path+data_label_aug[0]+'/pcd'+data_label_aug[0]+data_label_aug[1]+'mcrop.png'
        mask_rotated = mask_croped.rotate(rotate_angle)
        mask_rotated.save(path_maskrotated)
        #mask_rotated.show()

        # 3. load rectangle
        scale = 1
        rec_list = []
        path = dataset_path +data_label[0]+'/pcd'+data_label[0]+data_label[1]+'c'+'pos'+'.txt'
        for line in open(path).readlines():
            rec_str = line.split(' ')
            x = float(rec_str[0])
            y = float(rec_str[1])
            rec_list.append([x/scale, y/scale]) # zoom operation.
        rec_array = np.array(rec_list, dtype=np.float32).reshape(len(rec_list)/4, 8) #each rectangle represented by 4 points.

        # rotate rectangle
        path_rec_rotate = dataset_path +data_label_aug[0]+'/pcd'+data_label_aug[0]+data_label_aug[1]+'c'+'pos'+'.txt'
        #print('path_rec_rotate',path_rec_rotate)
        file2write=open(path_rec_rotate,'w')
        for n in xrange(len(rec_array)):
            rec = rec_array[n].tolist()
            rec_rotated = rotate_rectanle(rec, rotate_angle)
            #print('rec_rotated',rec_rotated)

            draw_rec2 = ImageDraw.Draw(img_rotated)
            draw_rec2.line((rec_rotated[0],rec_rotated[1])+(rec_rotated[2],rec_rotated[3]), fill='yellow', width=2)
            draw_rec2.line((rec_rotated[2],rec_rotated[3])+(rec_rotated[4],rec_rotated[5]), fill='green', width=2)
            draw_rec2.line((rec_rotated[4],rec_rotated[5])+(rec_rotated[6],rec_rotated[7]), fill='red', width=2)
            draw_rec2.line((rec_rotated[6],rec_rotated[7])+(rec_rotated[0],rec_rotated[1]), fill='blue', width=2)
            #img_rotated.show()
            for i in xrange(4):
                file2write.write(str(int(rec_rotated[2*i]))+' '+str(int(rec_rotated[2*i+1]))+'\n')
        file2write.close()


# label preparation:
def label_handling(data_label_1,data_label_2):
    '''data_label_1: file name, data_label_2: picture number '''
    data_label = []
    data_label_aug = []

    if data_label_1 < 10 :
        data_label.append(str(0)+str(data_label_1))
        data_label_aug.append(str(2)+str(data_label_1))
    else:
        data_label.append(str(data_label_1))
        data_label_aug.append(str(data_label_1+20))

    if data_label_2 < 10 :
        data_label.append(str(0)+str(data_label_2))
        data_label_aug.append(str(0)+str(data_label_2))
    else:
        data_label.append(str(data_label_2))
        data_label_aug.append(str(data_label_2))

    return data_label, data_label_aug

def rotate_rectanle(rec, angle):
    angle = -angle*(np.pi/180)
    #rotate_matrix = np.array([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0,0,1]])
    #print('rotate_matrix',rotate_matrix)
    rec_rotated = []
    for i in range(4): # each rectangle has 4 points.
        # origin points
        x = rec[2*i]
        y = rec[2*i+1]
        # rotate center point
        rx0 = 150
        ry0 = 150
        # aftrt rotate
        #print('np.cos(angle)',np.cos(angle), 'np.sin(angle)',np.sin(angle))
        x0 = (x - rx0)*np.cos(angle) - (y - ry0)*np.sin(angle) + rx0
        y0 = (x - rx0)*np.sin(angle) + (y - ry0)*np.cos(angle) + ry0
        rec_rotated.append(x0)
        rec_rotated.append(y0)
    return rec_rotated




    pass

if __name__ == '__main__':
    augment_multiple()
    #display_augment()
