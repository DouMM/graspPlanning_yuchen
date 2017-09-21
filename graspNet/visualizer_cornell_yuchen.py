# coding=utf-8

# python library
import numpy as np
from scipy import misc
from PIL import Image, ImageDraw, ImageFont
import shutil
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import json


# display: image, rectangle, img_croped, rect_croped, mask
def display_all():
    dataset_path = '/home/yuchen/catkin_ws/src/Robot_grasp/grasp_dataset/'
    file_num = 11
    img_num = 0
    data_label = label_handling(file_num, img_num)

    #---------------image and rectangle
    # load image
    path = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'r.png'
    #print('image path:', path)
    img = Image.open(path)
    print('image.size:', img.size)
    #print('image.format:', img.format)
    #print('image.mode:', img.mode)

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
    img.show()

    # ------------- img_croped and rect_croped
    path = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'crop.png'
    img_croped = Image.open(path)
    print('img_croped.size:', img_croped.size)
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
    img_croped.show()

    #------------------mask
    path_mask = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'m.png'
    mask = Image.open(path_mask)
    print('img_croped.size:', mask.size)
    mask.show()

    #-----------------mask_croped
    path_mcrop = dataset_path+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'mcrop.png'
    mask_croped = Image.open(path_mcrop)
    print('img_croped.size:', mask_croped.size)
    mask_croped.show()

#visualize loss reduction
def loss_visualizer():

    epoch = []
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    f = open('./result/log', 'r') #load log file
    data = json.load(f)
    f.close()

    value = []

    for i in range(0,len(data)):
        value = data[i]
        epoch.append(value["epoch"])
        train_loss.append(value["main/loss"])
        test_loss.append(value["validation/main/loss"])
        train_accuracy.append(value["main/accuracy"])
        test_accuracy.append(value["validation/main/accuracy"])

    #fig1 = plt.figure(1)
    fig1 = plt.figure(1,figsize=(8,6))
    plt.plot(epoch,train_loss,"b",linewidth=2,label = "train LOSS")
    plt.plot(epoch,test_loss,"g",linewidth=2,label = "validation LOSS")
    plt.yscale('log')
    #plt.title("LOSS reduction")
    plt.legend(fontsize=18)
    plt.xlabel("epoch",fontname='roman', fontsize=22)
    plt.ylabel("LOSS",fontname='roman', fontsize=22)
    plt.tick_params(labelsize=18)
    fig1.subplots_adjust(bottom=0.15)
    ax = fig1.add_subplot(111)

    #fig2 = plt.figure(2)
    fig2 = plt.figure(2,figsize=(8,6))
    plt.plot(epoch,train_accuracy,"b",linewidth=2,label = "train accuracy")
    plt.plot(epoch,test_accuracy,"g",linewidth=2,label = "validation accuracy ")
    #plt.title("accuracy increase")
    plt.legend(loc = "lower right",fontsize=18)
    plt.xlabel("epoch",fontname='roman',fontsize=22)
    plt.ylabel("accuracy",fontname='roman',fontsize=22)
    plt.yticks([i*0.1 for i in range(5,10,1)])
    plt.tick_params(labelsize=18)
    fig2.subplots_adjust(bottom=0.15)
    ax = fig2.add_subplot(111)

def draw_recPose_yuchen(img, rec_pose):
    # 1. image
    img_data = np.reshape(img,(120,160,3))
    img_data = np.uint8(img_data)
    img = Image.fromarray(img_data)
    resize_img = img.resize((img.size[0]*zoom,img.size[1]*zoom))
    #resize_img.show()

# draw rectangle
def draw_rec_yuchen(img, rec_grasp, img_height, img_width, img_channel):
    # 1. load img and rec.
    img_shape = img.shape # 32*32*3
    print('draw_rec_yuchen: img_shape', img_shape)

    # 2. change array to image and zoom it.
    #img_data = np.reshape(img,(120,160,3))
    img_data = np.reshape(img,(img_height,img_width,img_channel))
    img_data = np.uint8(img_data)
    img = Image.fromarray(img_data)
    zoom = 16
    resize_img = img.resize((img.size[0]*zoom,img.size[1]*zoom)) # zoom　image.
    resize_img.show()
    draw_rec = ImageDraw.Draw(resize_img)

    # 3. change grasp pose into rec coordinate
    print('draw_rec_yuchen: grasp', rec_grasp)
    xc = rec_grasp[0]
    yc = rec_grasp[1]
    theta = rec_grasp[2]
    a = rec_grasp[3]
    b = rec_grasp[4]

    rec = np.zeros(8)
    rec[0] = a*np.cos(theta)-b*np.sin(theta)+xc
    rec[1] = a*np.sin(theta)+b*np.cos(theta)+yc

    rec[2] = -a*np.cos(theta)-b*np.sin(theta)+xc
    rec[3] = -a*np.sin(theta)+b*np.cos(theta)+yc

    rec[4] = -a*np.cos(theta)+b*np.sin(theta)+xc
    rec[5] = -a*np.sin(theta)-b*np.cos(theta)+yc

    rec[6] = a*np.cos(theta)+b*np.sin(theta)+xc
    rec[7] = a*np.sin(theta)-b*np.cos(theta)+yc

    rec = rec*zoom
    draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='yellow', width=2)
    draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='green', width=2)
    draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='yellow', width=2)
    draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='green', width=2)
    actual_label = 'positive'

    image_label = 'img and rec'
    #draw = ImageDraw.Draw(image)
    draw_rec.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
    draw_rec.text((10,10), image_label, (255, 0, 0))

    resize_img.show()

# draw rectangle
def draw_rec(x, estimated,actual):

    zoom = 3
    img,rec = data_separator(x)
    img_shape = img.shape

    img_data = np.reshape(img,(120,160,3))
    img_data = np.uint8(img_data)
    img = Image.fromarray(img_data)
    resize_img = img.resize((img.size[0]*zoom,img.size[1]*zoom))
    #resize_img.show()

    rec = rec*zoom
    draw_rec = ImageDraw.Draw(resize_img)

    if actual == 0:
        draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='red', width=2)
        draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='blue', width=2)
        draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='red', width=2)
        draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='blue', width=2)
        actual_label = 'negative'
    elif actual == 1:
        draw_rec.line((rec[0],rec[1])+(rec[2],rec[3]), fill='yellow', width=2)
        draw_rec.line((rec[2],rec[3])+(rec[4],rec[5]), fill='green', width=2)
        draw_rec.line((rec[4],rec[5])+(rec[6],rec[7]), fill='yellow', width=2)
        draw_rec.line((rec[6],rec[7])+(rec[0],rec[1]), fill='green', width=2)
        actual_label = 'positive'

    #set image label
    if estimated == 0:
        estimated_label = 'negative'
    else:
        estimated_label = 'positive'

    image_label = " estimated: " + estimated_label + " actual: " + actual_label
    #draw = ImageDraw.Draw(image)
    draw_rec.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 20)
    draw_rec.text((10,10), image_label, (255, 0, 0))

    resize_img.show()

# separate input data into image and rec
def data_separator(X):

    img = []
    rec = []

    for i in range(len(X)):
        if i < 8:
            rec.append(X[i])
        else:
            img.append(X[i]*255)

    rec = np.asarray(rec).reshape(8,1).astype(np.float32)
    img = np.asarray(img).reshape(3,160,120).astype(np.float32)
    return img,rec

#load point cloud data
def load_point_cloud(data_label):

    image_label ='directly:' + str(data_label[0]) + ' picture:' + str(data_label[1])

    shutil.copy('../../grasp_dataset/'+data_label[0]+'/pcd'+data_label[0]+data_label[1]+'.txt','pcd_data'+data_label[0]+data_label[1]+'.pcd')
    point_cloud_raw = pcl.load('pcd_data'+data_label[0]+data_label[1]+'.pcd')
    os.remove('pcd_data'+data_label[0]+data_label[1]+'.pcd')
    point_cloud = np.asarray(point_cloud_raw)

    x = []
    y = []
    z = []

    for i in range(0,len(point_cloud),100):
        x.append(point_cloud[i][0])
        y.append(point_cloud[i][1])
        z.append(point_cloud[i][2])

    i = data_label[0] + '-' + data_label[1]

    fig = plt.figure(i)
    ax = Axes3D(fig)
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    ax.set_title(image_label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
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
#main
if __name__ == '__main__':
    display_all()
