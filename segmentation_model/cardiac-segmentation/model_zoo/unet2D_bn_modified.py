# -*- coding: utf-8 -*-
# Authors:
# Víctor M. Campello (vicmancr@gmail.com)


import tensorflow as tf
from tfwrapper import layers



def forward(images, training, nlabels):

    # images_padded = tf.pad(images, [[0,0], [92, 92], [92, 92], [0,0]], 'CONSTANT')

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64, training=training, padding='SAME')
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='SAME')

    pool1 = layers.max_pool_layer2d(conv1_2, 'pool_1')

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='SAME')

    pool2 = layers.max_pool_layer2d(conv2_2, 'pool_2')

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training, padding='SAME')
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='SAME')

    pool3 = layers.max_pool_layer2d(conv3_2, 'pool_3')

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training, padding='SAME')
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='SAME')

    pool4 = layers.max_pool_layer2d(conv4_2, 'pool_4')

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='SAME')

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], 'crop_concat_4', axis=3)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training, padding='SAME')
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='SAME')

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)

    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], 'crop_concat_3', axis=3)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training, padding='SAME')
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='SAME')

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], 'crop_concat_2', axis=3)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training, padding='SAME')
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='SAME')

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], 'crop_concat_1', axis=3)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='SAME')
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='SAME')

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, padding='SAME')

    return pred
