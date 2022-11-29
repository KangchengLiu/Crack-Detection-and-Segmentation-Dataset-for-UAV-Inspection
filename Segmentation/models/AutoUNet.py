import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers
import numpy as np


def Unet(inputs, num_classes):
    #inputs = Input((3, img_w, img_h))

    conv1 = slim.convolution2d(inputs,64, (3, 3))
    conv1 = slim.convolution2d(conv1,64, (3, 3))
    pool1 = slim.max_pool2d(conv1,(2,2))

    conv2 = slim.convolution2d(pool1,128, (3, 3))
    conv2 = slim.convolution2d(conv2,128, (3, 3))
    pool2 = slim.max_pool2d(conv2,(2,2))

    conv3 = slim.convolution2d(pool2,256, (3, 3))
    conv3 = slim.convolution2d(conv3,256, (3, 3))
    pool3 = slim.max_pool2d(conv3,(2,2))

    conv4 = slim.convolution2d(pool3,512, (3, 3))
    conv4 = slim.convolution2d(conv4,512, (3, 3))
    pool4 = slim.max_pool2d(conv4,(2,2))

    conv5 = slim.convolution2d(pool4,1024, (3, 3))
    conv5 = slim.convolution2d(conv5,1024, (3, 3))
    
    up6 = tf.concat([slim.convolution2d_transpose(conv5,1024,(3, 3),2), conv4], axis=-1)
    conv6 = slim.convolution2d(up6,512, (3,3))
    conv6 = slim.convolution2d(conv6,512, (3,3))

    up7 = tf.concat([slim.convolution2d_transpose(conv6,512,(3, 3),2), conv3], axis=-1)
    conv7 = slim.convolution2d(up7,256, (3,3))
    conv7 = slim.convolution2d(conv7,256, (3,3))

    up8 = tf.concat([slim.convolution2d_transpose(conv7,256,(3, 3),2), conv2], axis=-1)
    conv8 = slim.convolution2d(up8,128,(3,3))
    conv8 = slim.convolution2d(conv8,128,(3,3))
    
    up9 = tf.concat([slim.convolution2d_transpose(conv8,128,(3, 3),2), conv1], axis=-1)
    conv9 = slim.convolution2d(up9,64,(3,3))
    conv9 = slim.convolution2d(conv9,64,(3,3))

    conv10 = slim.convolution2d(conv9,num_classes,(1,1),activation_fn=None)
    
    return conv10
	
	
	

