# -*- coding: utf-8 -*-

import tensorflow as tf
from tfwrapper import layers



def forward(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64, training=training, padding='SAME')
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training, padding='SAME')

    pool1 = layers.max_pool_layer2d(conv1_2, 'pool_1')

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training, padding='SAME')
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training, padding='SAME')

    pool2 = layers.max_pool_layer2d(conv2_2, 'pool_2')
    dout2 = layers.dropout_layer(pool2, 'dropout_2', training)

    conv3_1 = layers.conv2D_layer_bn(dout2, 'conv3_1', num_filters=256, training=training, padding='SAME')
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training, padding='SAME')

    pool3 = layers.max_pool_layer2d(conv3_2, 'pool_3')
    dout3 = layers.dropout_layer(pool3, 'dropout_3', training)

    conv4_1 = layers.conv2D_layer_bn(dout3, 'conv4_1', num_filters=512, training=training, padding='SAME')
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training, padding='SAME')

    pool4 = layers.max_pool_layer2d(conv4_2, 'pool_4')
    dout4 = layers.dropout_layer(pool4, 'dropout_4', training)

    conv5_1 = layers.conv2D_layer_bn(dout4, 'conv5_1', num_filters=1024, training=training, padding='SAME')
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training, padding='SAME')

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], 'crop_concat_4', axis=3)
    dout5 = layers.dropout_layer(concat4, 'dropout_5', training)

    conv6_1 = layers.conv2D_layer_bn(dout5, 'conv6_1', num_filters=512, training=training, padding='SAME')
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training, padding='SAME')

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], 'crop_concat_3', axis=3)
    dout6 = layers.dropout_layer(concat3, 'dropout_6', training)

    conv7_1 = layers.conv2D_layer_bn(dout6, 'conv7_1', num_filters=256, training=training, padding='SAME')
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training, padding='SAME')

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], 'crop_concat_2', axis=3)
    dout7 = layers.dropout_layer(concat2, 'dropout_7', training)

    conv8_1 = layers.conv2D_layer_bn(dout7, 'conv8_1', num_filters=128, training=training, padding='SAME')
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training, padding='SAME')

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
    concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], 'crop_concat_1', axis=3)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training, padding='SAME')
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training, padding='SAME')

    pred_1 = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, padding='SAME')

    # Deep supervision
    ds1_1 = layers.conv2D_layer(conv7_2, 'ds_1', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, padding='SAME')
    ds1_2 = layers.deconv2D_layer(ds1_1, 'ds_2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, padding='SAME')
    ds2_1 = layers.conv2D_layer(conv8_2, 'ds_3', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, padding='SAME')
    ds1_ds2 = tf.add(ds1_2, ds2_1)
    ds = layers.deconv2D_layer(ds1_ds2, 'ds_4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, padding='SAME')

    pred_2 = tf.add(pred_1, ds)

    return pred_2
