# -*- coding: utf-8 -*-
import tensorflow as tf
from inspect import signature

from tfwrapper import losses, layers


class Model(object):
    '''
    Class for training a deep learning model. It must implement the basic methods.
    '''

    def __init__(self, exp_config):
        '''
        Constructor. It sets the configuration for the whole process.
        :param exp_config: (dict) Configuration for the experiment.
        '''
        # Get number of parameters needed for model handle
        self.config = exp_config
        sig = signature(self.config.model_handle)
        self.params = len(sig.parameters) - 1


    def initialize(self):
        '''
        Creates main tensors to use during all processes.
        :param exp_config: (dict) Configuration for the experiment.
        '''
        image_tensor_shape = [self.config.batch_size] + list(self.config.image_size) + [1]
        mask_tensor_shape  = [self.config.batch_size] + list(self.config.label_size)

        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        labels_pl = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name='labels')

        learning_rate_pl = tf.placeholder(tf.float32, shape=[])
        training_pl = tf.placeholder(tf.bool, shape=[])

        return images_pl, labels_pl, learning_rate_pl, training_pl


    def inference(self, *args):
        '''
        Wrapper function to provide an interface to a model from the model_zoo inside of the model module.
        args: images, training, labels
        '''
        return self.config.model_handle(*args[:self.params], self.config.nlabels)


    def loss(self, logits, labels, images):
        '''
        Loss to be minimised by the neural network
        :param logits: The output of the neural network before the softmax
        :param labels: The ground truth labels in standard (i.e. not one-hot) format
        :param images: The input
        :return: The total loss including weight decay, the loss without weight decay, only the weight decay 
        '''
        nlabels = self.config.nlabels
        loss_type = self.config.loss_type
        weight_decay = self.config.weight_decay
        loss_hyper_params = self.config.loss_hyper_params if hasattr(self.config, 'loss_hyper_param') else [1, 0.2]

        if nlabels > 2:
            oh_labels = tf.one_hot(labels, depth=nlabels)
        else:
            oh_labels = tf.cast(labels, tf.float32)

        with tf.variable_scope('weights_norm'):

            weights_norm = tf.reduce_sum(
                input_tensor = weight_decay*tf.stack(
                    [tf.nn.l2_loss(ii) for ii in tf.get_collection('weight_variables')]
                ),
                name='weights_norm'
            )

        if loss_type == 'weighted_crossentropy':
            segmentation_loss = losses.pixel_wise_cross_entropy_loss_weighted(logits, oh_labels,
                                                                            class_weights=[0.1, 0.3, 0.3, 0.3])
        elif loss_type == 'crossentropy':
            segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, oh_labels)
        elif loss_type == 'dice':
            segmentation_loss = losses.dice_loss(logits, oh_labels, only_foreground=False)
        elif loss_type == 'dice_onlyfg':
            segmentation_loss = losses.dice_loss(logits, oh_labels, only_foreground=True)
        elif loss_type == 'crossentropy_and_dice':
            segmentation_loss = loss_hyper_params[0] * losses.pixel_wise_cross_entropy_loss(logits, oh_labels) \
                + loss_hyper_params[1] * losses.dice_loss(logits, oh_labels)
        elif loss_type == 'dice_cc':
            segmentation_loss = loss_hyper_params[0] * losses.dice_loss(logits, oh_labels) \
                + loss_hyper_params[1] * losses.connected_component_loss(logits, oh_labels)
        else:
            raise ValueError('Unknown loss: %s' % loss_type)


        total_loss = tf.add(segmentation_loss, weights_norm)

        return total_loss, segmentation_loss, weights_norm


    def predict(self, images):
        '''
        Returns the prediction for an image given a network from the model zoo
        :param images: An input image tensor
        :return: A prediction mask, and the corresponding softmax output
        '''

        if self.params > 2:
            logits = self.config.model_handle(images, training=tf.constant(False, dtype=tf.bool), labels=tf.constant(0), nlabels=self.config.nlabels)
            softmax = tf.nn.softmax(logits[0])
            mask = tf.argmax(softmax, axis=-1)        
        else:
            logits = self.config.model_handle(images, training=tf.constant(False, dtype=tf.bool), nlabels=self.config.nlabels)
            softmax = tf.nn.softmax(logits)
            mask = tf.argmax(softmax, axis=-1)

        return mask, softmax


    def training_step(self, loss, learning_rate):
        '''
        Creates the optimisation operation which is executed in each training iteration of the network
        :param loss: The loss to be minimised
        :param learning_rate: Learning rate 
        :return: The training operation
        '''
        optimizer_handle = self.config.optimizer_handle
        if self.config.momentum is not None:
            optimizer = optimizer_handle(learning_rate=learning_rate, momentum=self.config.momentum)
        else:
            optimizer = optimizer_handle(learning_rate=learning_rate)

        # The with statement is needed to make sure the tf contrib version of batch norm properly performs its updates
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)

        return train_op


    def evaluation(self, logits, labels, images):
        '''
        A function for evaluating the performance of the netwrok on a minibatch. This function returns the loss and the 
        current foreground Dice score, and also writes example segmentations and imges to to tensorboard.
        :param logits: Output of network before softmax
        :param labels: Ground-truth label mask
        :param images: Input image mini batch
        :return: The loss without weight decay, the foreground dice of a minibatch
        '''
        nlabels = self.config.nlabels

        if self.params > 2:
            output = logits[0]
        else:
            output = logits

        mask = tf.argmax(tf.nn.softmax(output, axis=-1), axis=-1)  # reduce dimensionality
        mask_gt = labels

        tf.summary.image('example_gt', self.prepare_tensor_for_summary(mask_gt, mode='mask', nlabels=nlabels))
        tf.summary.image('example_pred', self.prepare_tensor_for_summary(mask, mode='mask', nlabels=nlabels))
        tf.summary.image('example_zimg', self.prepare_tensor_for_summary(images, mode='image'))

        _, nowd_loss, _ = self.loss(logits, labels, images)

        cdice_structures = losses.per_structure_dice(output, tf.one_hot(labels, depth=nlabels))
        cdice_foreground = cdice_structures[:,1:]

        cdice = tf.reduce_mean(cdice_foreground)
        cdice_batch = tf.reduce_mean(cdice_foreground, axis=-1)

        return nowd_loss, cdice, cdice_batch


    def prepare_tensor_for_summary(self, img, mode, idx=0, nlabels=None):
        '''
        Format a tensor containing imgaes or segmentation masks such that it can be used with
        tf.summary.image(...) and displayed in tensorboard. 
        :param img: Input image or segmentation mask
        :param mode: Can be either 'image' or 'mask. The two require slightly different slicing
        :param idx: Which index of a minibatch to display. By default it's always the first
        :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label.. 
        :return: Tensor ready to be used with tf.summary.image(...)
        '''
        img_w = tf.shape(img)[1]
        img_h = tf.shape(img)[2]

        if mode == 'mask':

            if img.get_shape().ndims == 3:
                V = img[idx, ...]
            elif img.get_shape().ndims == 4:
                V = tf.squeeze(img[idx, ...])
            elif img.get_shape().ndims == 5:
                V = img[idx, ..., 4, 0]
            else:
                raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

        elif mode == 'image':

            if img.get_shape().ndims == 3:
                V = img[idx, ...]
            elif img.get_shape().ndims == 4:
                V = tf.squeeze(img[idx, ...])
            elif img.get_shape().ndims == 5:
                V = img[idx, ..., 4, 0]
            else:
                raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

        else:
            raise ValueError('Unknown mode: %s. Must be image or mask' % mode)

        if mode == 'image' or not nlabels:
            V -= tf.reduce_min(V)
            V /= tf.reduce_max(V)
        else:
            V /= (nlabels - 1)  # The largest value in a label map is nlabels - 1.

        V *= 255
        V = tf.cast(V, dtype=tf.uint8)

        V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
        return V
