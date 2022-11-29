from __future__ import print_function, division
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, random
#from scipy.misc import imread
from imageio import imread
import ast
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

import helpers
from keras import backend as K

from enum import Enum
from scipy.ndimage import distance_transform_edt as distance
import tensorflow.keras as K

from tensorflow.python.ops import array_ops

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


# Count total number of parameters in the model
def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

# Subtracts the mean images from ImageNet
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def _flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def _lovasz_softmax_flat(probas, labels, only_present=True):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.shape[1]
    losses = []
    present = []
    for c in range(C):
        fg = tf.cast(tf.equal(labels, c), probas.dtype) # foreground for class c
        if only_present:
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    losses_tensor = tf.stack(losses)
    if only_present:
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    return losses_tensor

def lovasz_softmax(probas, labels, only_present=True, per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    probas = tf.nn.softmax(probas, 3)
    labels = helpers.reverse_one_hot(labels)

    if per_image:
        def treat_image(prob, lab):
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = _flatten_probas(prob, lab, ignore, order)
            return _lovasz_softmax_flat(prob, lab, only_present=only_present)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
    else:
        losses = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore, order), only_present=only_present)
    return losses


# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape exceeds image dimensions!')

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def compute_mean_iou(pred, label):

    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou

    
def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes) 

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)

            
        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights

# Compute the memory usage, for debugging
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)


'''
Compatible with tensorflow backend
'''

def binary_crossentropy(probas,labels):
    #loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true = labels, y_pred = probas, from_logits = False))
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true = labels, y_pred = probas))
    return loss

def dice_score(probas,labels):
    smooth = K.backend.epsilon()
    y_true_f = K.backend.flatten(labels)
    y_pred_f = K.backend.flatten(probas)
    intersection = K.backend.sum(y_true_f * y_pred_f)
    answer = (2. * intersection + smooth) / (K.backend.sum(y_true_f) + K.backend.sum(y_pred_f) + smooth)
    return answer

def dice_loss(probas,labels):
    answer = 1. - dice_score(labels, probas)
    return answer


def cross_and_dice_loss(probas,labels):
    cross_entropy_value = binary_crossentropy(labels, probas)
    dice_loss_value = dice_loss(labels, probas)
    return 0.5 * dice_loss_value + 0.5 * cross_entropy_value

'''def focal_loss(pred,target,weight=None,gamma=2.0,alpha=0.25,reduction='mean',avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1-pred_sigmoid)*target+pred_sigmoid*(1-target)
    focal_weight = (alpha*target+(1-alpha)*(1-target))*pt.pow(gamma)
    loss = F.binary_cross_entroy_with_logits(pred,target,reduction='none')*focal_weight
    loss = weight_reduce_loss(loss,weight,reductioin,avg_factor)
    return loss

def focal_loss(y_pred,y_true,gamma=2.0,alpha=0.25):
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    predictions = tf.sigmoid(y_pred)
    predictions_pt = tf.where(tf.equal(y_true,1),predictions,1.-predictions)

    alpha_t = tf.scalar_mul(alpha,tf.ones_like(y_pred,dtype=tf.float32))
    alpha_t = tf.where(tf.equal(y_true,1.0),alpha_t,1-alpha_t)
    weighted_loss = ce * tf.pow(1-predictions_pt,gamma) * alpha_t
    loss = tf.reduce_sum(weighted_loss)

    return loss'''

def Focal_loss(y_pred,y_true,alpha=0.25,gamma = 2):
    pt_1 = tf.where(tf.equal(y_true,1),y_pred,tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true,0),y_pred,tf.zeros_like(y_pred))

    #原损失值的计算
    loss =  -K.backend.mean(alpha * K.backend.pow(1. - pt_1,gamma) * K.backend.log(pt_1)) - K.backend.mean((1-alpha) * K.backend.pow(pt_0,gamma) * K.backend.log(1. -pt_0))
    #参考文章进行的修改
    #loss =  -K.backend.mean(alpha * y_true * K.backend.pow(1. - y_pred,gamma) * K.backend.log(y_pred)) - K.backend.mean((1-alpha) * y_true * K.backend.pow(y_pred,gamma) * K.backend.log(1. -y_pred))
    
    return loss

def focal_loss_sigmoid(probas,labels,gamma=2.0,alpha=0.25):
    Y_pred = tf.nn.sigmoid(probas)
    Labels = tf.to_float(labels)
    L = -Labels*(1-alpha)*((1-Y_pred)*gamma)*tf.log(Y_pred+10e-10)-(1-Labels)*alpha*(Y_pred**gamma)*tf.log(1-Y_pred+10e-10)
    return L

'''def focal_loss(y_pred,y_true,weights=None,alpha=0.25,gamma=2):
    sigmoid_p = tf.nn.sigmoid(y_pred)
    zeros = array_ops.zeros_like(sigmoid_p,dtype=sigmoid_p.dtype)

    pos_p_sub = array_ops.where(y_true > zeros,y_true - sigmoid_p,zeros)

    neg_p_sub = array_ops.where(y_true > zeros, zeros, sigmoid_p)

    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8,1.0))- (1-alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p,1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)'''

def focal_loss(gamma=2,alpha=0.75):
    def focal_loss_fixed(probas,labels):
        eps = 1e-12
        y_pred = K.clip(probas,eps,1.-eps)
        pt_1 = tf.where(tf.equal(labels,1),y_pred,tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(labels,0),y_pred,tf.zeros_like(y_pred))
        #return -K.sum(alpha*K.pow(1.-pt_1,gamma)*K.log(pt_1))-K.sum((1-alpha)*K.pow(pt_0,gamma)*K.log(1.-pt_0))
        return -K.mean(alpha*K.pow(1.-pt_1,gamma)*K.log(pt_1 + 10e-10))-K.mean((1-alpha)*K.pow(pt_0,gamma)*K.log(1.-pt_0 + 10e-10))
    return focal_loss_fixed

