# coding=utf-8
#!/usr/bin/env python
# python library
import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw, ImageFont
# OpenCV
import cv2

def crop_singleimage():
    dataset_path = '/informatik2/tams/home/deng/catkin_ws/src/Robot_grasp/grasp_dataset/'
    file_num = 1
    img_num = 0
    data_label = label_handling(file_num, img_num)

    # load image
    path = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'
    path_crop = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'crop.png'
    #print('image path:', path)
    img = Image.open(path)
    print('image.size:', img.size)
    #print('image.format:', img.format)
    #print('image.mode:', img.mode)

    # change pixel
    box = (90, 160, 390, 460)
    img_croped = img.crop(box)
    #img_croped.show()

    # load rectangle for img
    scale = 1
    rec_list = []
    path = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'c'+'pos'+'.txt'
    for line in open(path).readlines():
        rec_str = line.split(' ')
        x = float(rec_str[0])
        y = float(rec_str[1])
        rec_list.append([x/scale,y/scale])
    rec_array = np.array(rec_list, dtype=np.float32).reshape(len(rec_list)/4, 8) #
    # plot image and rectangle.
    rec = rec_array[0].tolist()
    draw_rec = ImageDraw.Draw(img)
    draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='yellow', width=2)
    draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='green', width=2)
    draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='red', width=2)
    draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='blue', width=2)
    #img.show()

    # load rectangle for img_croped
    rec = rec_array[0].tolist()
    # coordinate adjust:
    rec[0] = rec[0] - 90
    rec[1] = rec[1] - 160
    rec[2] = rec[2] - 90
    rec[3] = rec[3] - 160
    rec[4] = rec[4] - 90
    rec[5] = rec[5] - 160
    rec[6] = rec[6] - 90
    rec[7] = rec[7] - 160
    draw_rec = ImageDraw.Draw(img_croped)
    draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='yellow', width=2)
    draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='green', width=2)
    draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='red', width=2)
    draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='blue', width=2)
    #img_croped.show()
    img_croped.save(path_crop)
    img_croped.close()

def crop_allimage():
    dataset_path_1 = '/informatik2/tams/home/deng/catkin_ws/src/Robot_grasp/tmp/'
    dataset_path = '/informatik2/tams/home/deng/catkin_ws/src/Robot_grasp/grasp_dataset/'
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
        data_label = label_handling(file_num, img_num)
        # load image
        path = dataset_path_1+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'
        path_crop = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'crop.png'
        #print('image path:', path)
        img = Image.open(path)
        box = (90, 160, 390, 460)
        img_croped = img.crop(box)
        img_croped.save(path_crop)
        #img_croped.show()

        # load mask
        path_mask = dataset_path_1+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'m.png'
        path_maskcrop = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'mcrop.png'
        #print('image path:', path)
        mask = Image.open(path_mask)
        box = (90, 160, 390, 460)
        mask_croped = mask.crop(box)
        mask_croped.save(path_maskcrop)
        #mask_croped.show()

        # load rectangle
        scale = 1
        rec_list = []
        path = dataset_path_1 +data_label[0]+'/pcd'+data_label[0]+data_label[1]+'c'+'pos'+'.txt'
        for line in open(path).readlines():
            rec_str = line.split(' ')
            x = float(rec_str[0]) - 90 #crop operation.
            y = float(rec_str[1]) - 160
            rec_list.append([x/scale, y/scale]) # zoom operation.
        rec_array = np.array(rec_list, dtype=np.float32).reshape(len(rec_list)/4, 8) #each rectangle represented by 4 points.

        # rotate rectangle
        path_rec_rotate = dataset_path +data_label[0]+'/pcd'+data_label[0]+data_label[1]+'c'+'pos'+'.txt'
        #print('path_rec_rotate',path_rec_rotate)
        file2write=open(path_rec_rotate,'w')
        for n in xrange(len(rec_array)):
            rec = rec_array[n].tolist()
            draw_rec = ImageDraw.Draw(img_croped)
            draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='yellow', width=2)
            draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='green', width=2)
            draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='red', width=2)
            draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='blue', width=2)
            #img_croped.show()
            for i in xrange(4):
                file2write.write(str(int(rec[2*i]))+' '+str(int(rec[2*i+1]))+'\n')
        file2write.close()



# label preparation:
def label_handling(data_label_1,data_label_2):
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

if __name__ == '__main__':
    crop_allimage()
    #ROI_read_single()
