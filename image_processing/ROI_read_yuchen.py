# coding=utf-8
#!/usr/bin/env python
# python library
import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw, ImageFont
# OpenCV
import cv2

def ROI_read_multipleRec():
    dataset_path = '/home/yuchen/catkin_ws/src/Robot_grasp/grasp_dataset/'
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
        path = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'
        path_mast = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'m.png'
        #print('image path:', path)
        img = Image.open(path)
        #draw_rec = ImageDraw.Draw(img)
        #print('image.size:', img.size)
        #print('image.format:', img.format)
        #print('image.mode:', img.mode)

        # laod rectangle
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
        img = img.convert('L')
        pixelmap_img = img.load()
        for i in range(img.size[0]):    # for every col:
            for j in range(img.size[1]):    # For every row
                pixelmap_img[i,j] = 0
        for n in range(len(rec_array)): # 3. diffrent positive grasp
            rec = rec_array[n].tolist()
            '''
            draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='yellow', width=2)
            draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='green', width=2)
            draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='red', width=2)
            draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='blue', width=2)
            #img.show()
            '''
            # change pixel
            point_x = np.array([int(rec[0]), int(rec[2]), int(rec[4]),int(rec[6])])
            point_y = np.array([int(rec[1]), int(rec[3]), int(rec[5]),int(rec[7])])
            point_x_sort = sorted(point_x)
            point_y_sort = sorted(point_y)
            #print('point_x', point_x, 'point_y', point_y)
            #print(np.argmax(point_x))
            point_right = [rec[np.argmax(point_x)*2], rec[np.argmax(point_x)*2+1]]
            #print('point_right', point_right)
            #print(np.nanargmin(point_x))
            point_left = [rec[np.nanargmin(point_x)*2], rec[np.nanargmin(point_x)*2+1]]
            #print(np.argmax(point_y))
            point_up = [rec[np.argmax(point_y)*2], rec[np.argmax(point_y)*2+1]]
            #print(np.nanargmin(point_y))
            point_down  = [rec[np.nanargmin(point_y)*2], rec[np.nanargmin(point_y)*2+1]]
            #print(point_left, point_right, point_up, point_down)

            for i in range(img.size[0]):    # for every col:
                for j in range(img.size[1]):    # For every row
                    if point_down[0] < point_up[0]:
                        if i>=point_left[0] and i<=point_down[0]:
                            line_up = line_function(i, point_left, point_up)
                            line_down = line_function(i, point_left, point_down)
                            if j >= line_down and j <= line_up:
                                pixelmap_img[i,j] = 255
                        elif i>=point_down[0] and i<=point_up[0]:
                            line_up = line_function(i, point_left, point_up)
                            line_down = line_function(i, point_down, point_right)
                            if j >= line_down and j <= line_up:
                                pixelmap_img[i,j] = 255
                        elif i>=point_up[0] and i<=point_right[0]:
                            line_up = line_function(i, point_up, point_right)
                            line_down = line_function(i, point_down, point_right)
                            if j >= line_down and j <= line_up:
                                pixelmap_img[i,j] = 255
                    else:
                        if i>=point_left[0] and i<=point_up[0]:
                            line_up = line_function(i, point_left, point_up)
                            line_down = line_function(i, point_left, point_down)
                            if j >= line_down and j <= line_up:
                                pixelmap_img[i,j] = 255
                        elif i>=point_up[0] and i<=point_down[0]:
                            line_up = line_function(i, point_up, point_right)
                            line_down = line_function(i, point_down, point_left)
                            if j >= line_down and j <= line_up:
                                pixelmap_img[i,j] = 255
                        elif i>=point_down[0] and i<=point_right[0]:
                            line_up = line_function(i, point_up, point_right)
                            line_down = line_function(i, point_down, point_right)
                            if j >= line_down and j <= line_up:
                                pixelmap_img[i,j] = 255
        img.save(path_mast)
        img.close()
        #img.show()

def ROI_read_singleRec():
    dataset_path = '/home/yuchen/catkin_ws/src/Robot_grasp/grasp_dataset/'
    file_num = 1
    img_num = 0
    data_label = label_handling(file_num, img_num)

    # load image
    path = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'
    #print('image path:', path)
    img = Image.open(path)
    draw_rec = ImageDraw.Draw(img)
    #print('image.size:', img.size)
    #print('image.format:', img.format)
    #print('image.mode:', img.mode)

    # laod rectangle
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
    draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='yellow', width=2)
    draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='green', width=2)
    draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='red', width=2)
    draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='blue', width=2)
    img.show()

    # change pixel
    point_x = np.array([int(rec[0]), int(rec[2]), int(rec[4]),int(rec[6])])
    point_y = np.array([int(rec[1]), int(rec[3]), int(rec[5]),int(rec[7])])
    point_x_sort = sorted(point_x)
    point_y_sort = sorted(point_y)
    #print('point_x', point_x, 'point_y', point_y)
    #print(np.argmax(point_x))
    point_right = [rec[np.argmax(point_x)*2], rec[np.argmax(point_x)*2+1]]
    #print('point_right', point_right)
    #print(np.nanargmin(point_x))
    point_left = [rec[np.nanargmin(point_x)*2], rec[np.nanargmin(point_x)*2+1]]
    #print(np.argmax(point_y))
    point_up = [rec[np.argmax(point_y)*2], rec[np.argmax(point_y)*2+1]]
    #print(np.nanargmin(point_y))
    point_down  = [rec[np.nanargmin(point_y)*2], rec[np.nanargmin(point_y)*2+1]]
    print(point_left, point_right, point_up, point_down)

    img = img.convert('L')
    pixelmap_img = img.load()
    for i in range(img.size[0]):    # for every col:
        for j in range(img.size[1]):    # For every row
            if i > point_x_sort[0] and i < point_x_sort[-1] and j > point_y_sort[0] and j < point_y_sort[-1]:
                pixelmap_img[i,j] = 255 # set the colour accordingly
                if i >= point_left[0] and i <= point_down[0] and j >= point_down[1] and j <= point_left[1]:
                    #print('zone: left-down')
                    if line_function(i, point_down, point_left) >= j:
                        pixelmap_img[i,j] = 0
                if i >= point_down[0] and i <= point_right[0] and j >= point_down[1] and j <= point_right[1]:
                    #print('zone: right-down')
                    if line_function(i, point_down, point_right) >= j:
                        pixelmap_img[i,j] = 0
                if i >= point_left[0] and i <= point_up[0] and j >= point_left[1] and j <= point_up[1]:
                    #print('zone: right-down')
                    if line_function(i, point_left, point_up) <= j:
                        pixelmap_img[i,j] = 0
                if i >= point_up[0] and i <= point_right[0] and j >= point_right[1] and j <= point_up[1]:
                    #print('zone: right-down')
                    if line_function(i, point_up, point_right) <= j:
                        pixelmap_img[i,j] = 0
            else:
                pixelmap_img[i,j] = 0
    img.show()

def line_function(x, point_1, point_2):
    #print((point_2[1]- point_1[1])/(point_2[0]- point_1[0]))
    if point_2[0] == point_1[0]:
        subtract = 0.001
    else:
        subtract = point_2[0]- point_1[0]
    y = ((point_2[1]- point_1[1])/(subtract))*(x - point_1[0]) + point_1[1]
    return y

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
    ROI_read_multiple()
    #ROI_read_single()
