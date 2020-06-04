# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import importlib
import argparse
import logging
import time
import h5py
import sys
import cv2
import os
import re

import config.system as sys_config

from utils import utils_gen, image_utils
from utils.background_generator import BackgroundGenerator


def run_training(Model, exp_config, dataset, subset):
    '''
    Function that handles the training process.
    Parameters:
        exp_config: (dict) Configuration of selected experiment to perform.
        dataset: (str) Name of dataset folder to use.
    Return:
        Nothing.
    '''

    log_dir = os.path.join(sys_config.log_root, exp_config.experiment_name)
    # Data information
    mode = exp_config.data_mode
    subset = '_' + subset
    size_str = '_'.join([str(i) for i in exp_config.image_size])
    res_str = '_'.join([str(i) for i in exp_config.target_resolution])
    suffix = '' if exp_config.train_on_all_data else '_onlytrain'
    data_file_name = exp_config.data_file_mask.format(dataset, mode, subset, size_str, res_str, suffix)

    continue_run = True
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        continue_run = False

    # Copy experiment config file
    with open(exp_config.__file__, 'r') as f:
        lines = f.readlines()
    with open(os.path.join(log_dir, exp_config.experiment_name + '.py'), 'w') as f:
        for l in lines:
            if '=' in l:
                if isinstance(exp_config.__dict__[l.split()[0]], str):
                    f.write("{0} = '{1}'\n".format(l.split()[0], exp_config.__dict__[l.split()[0]]))
                    if l.split()[0] == 'data_file_mask':
                        f.write("data_file_subs = '{0}'\n".format(data_file_name))
                elif l.split()[0] in ['optimizer_handle', 'model_handle']:
                    f.write(l)
                else:
                    f.write("{0} = {1}\n".format(l.split()[0], exp_config.__dict__[l.split()[0]]))
            else:
                f.write(l)


    ######################################################################
    # Load previous model if exists.
    ######################################################################
    logging.info('EXPERIMENT NAME: {}'.format(exp_config.experiment_name))

    init_step = 0

    if continue_run:
        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!! Continuing previous run !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        try:
            init_checkpoint_path = utils_gen.get_latest_model_checkpoint_path(log_dir, 'model.ckpt')
            logging.info('Checkpoint path: {}'.format(init_checkpoint_path))
            init_step = int(init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 b/c otherwise starts with eval
            logging.info('Latest step was: {}'.format(init_step))
        except:
            logging.warning('!!! Did not find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_step = 0

        logging.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    ######################################################################


    ######################################################################
    # Load data for training.
    ######################################################################
    data = h5py.File(os.path.join(sys_config.data_base, dataset, 'preproc_data', data_file_name), 'r')

    # the following are HDF5 datasets, not numpy arrays
    images_train = data['data_train']
    labels_train = data['pred_train']

    if not exp_config.train_on_all_data:
        images_val = data['data_test']
        labels_val = data['pred_test']

    if exp_config.use_data_fraction:
        num_images = images_train.shape[0]
        new_last_index = int(float(num_images)*exp_config.use_data_fraction)

        logging.warning('USING ONLY FRACTION OF DATA!')
        logging.warning(' - Number of imgs orig: {0}, Number of imgs new: {1}'.format(num_images, new_last_index))
        images_train = images_train[0:new_last_index,...]
        labels_train = labels_train[0:new_last_index,...]

    logging.info('Data summary:')
    logging.info(' - Images:')
    logging.info(images_train.shape)
    logging.info(images_train.dtype)
    logging.info(' - Labels:')
    logging.info(labels_train.shape)
    logging.info(labels_train.dtype)
    ######################################################################


    ######################################################################
    # Initialize and define model.
    ######################################################################
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():

        ##################
        # Initialization #  Generate placeholders for the images and labels.
        ##################
        model = Model(exp_config)
        x_placeholder, y_placeholder, learning_rate_pl, training_pl = model.initialize()
        tf.summary.scalar('learning_rate', learning_rate_pl)

        #############
        # Inference #    Build a Graph that computes predictions from the inference model.
        #############
        logits = model.inference(x_placeholder, training_pl, y_placeholder)

        ########
        # Loss #    Add to the Graph the Ops for loss calculation (second output is unregularised loss).
        ########
        loss, _, weights_norm = model.loss(logits, y_placeholder, x_placeholder)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('weights_norm_term', weights_norm)

        ############
        # Training #    Add to the Graph the Ops that calculate and apply gradients.
        ############
        train_op = model.training_step(loss, learning_rate_pl)

        ##############
        # Evaluation #  Add the Op to compare the logits to the labels during evaluation.
        ##############
        eval_loss = model.evaluation(logits, y_placeholder, x_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        max_to_keep = 3
        saver = tf.train.Saver(max_to_keep=max_to_keep)
        saver_best_dice = tf.train.Saver()
        saver_best_xent = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
        config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
        sess = tf.Session(config=config)

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # with tf.name_scope('monitoring'):

        val_error_ = tf.placeholder(tf.float32, shape=[], name='val_error')
        val_error_summary = tf.summary.scalar('validation_loss', val_error_)

        val_dice_ = tf.placeholder(tf.float32, shape=[], name='val_dice')
        val_dice_summary = tf.summary.scalar('validation_dice', val_dice_)

        val_summary = tf.summary.merge([val_error_summary,
                                        val_dice_summary])

        train_error_ = tf.placeholder(tf.float32, shape=[], name='train_error')
        train_error_summary = tf.summary.scalar('training_loss', train_error_)

        train_dice_ = tf.placeholder(tf.float32, shape=[], name='train_dice')
        train_dice_summary = tf.summary.scalar('training_dice', train_dice_)

        train_summary = tf.summary.merge([train_error_summary,
                                          train_dice_summary])

        # Run the Op to initialize the variables.
        sess.run(init)

        step = init_step
        curr_lr = exp_config.learning_rate

        no_improvement_counter = 0
        best_val = np.inf
        last_train = np.inf
        loss_history = []
        loss_gradient = np.inf
        best_dice = 0

        if continue_run:
            # Restore session
            saver.restore(sess, init_checkpoint_path)
            if not exp_config.train_on_all_data:
                logging.info('Validation Data Eval:')
                [val_loss, val_dice] = do_eval(exp_config,
                                                sess,
                                                eval_loss,
                                                x_placeholder,
                                                y_placeholder,
                                                training_pl,
                                                images_val,
                                                labels_val,
                                                exp_config.batch_size,
                                                subset)
                best_dice = val_dice
                best_val = val_loss

        for epoch in range(exp_config.max_epochs):

            logging.info('EPOCH {}'.format(epoch))

            for batch in iterate_minibatches(exp_config, images_train,
                                             labels_train,
                                             batch_size=exp_config.batch_size,
                                             augment_batch=exp_config.augment_batch):

                if exp_config.warmup_training:
                    if step < 50:
                        curr_lr = exp_config.learning_rate / 10.0
                    elif step == 50:
                        curr_lr = exp_config.learning_rate

                start_time = time.time()

                x, y = batch

                # TEMPORARY HACK (to avoid incomplete batches)
                if y.shape[0] < exp_config.batch_size:
                    step += 1
                    continue

                feed_dict = {
                    x_placeholder: x,
                    y_placeholder: y,
                    learning_rate_pl: curr_lr,
                    training_pl: True
                }


                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                duration = time.time() - start_time

                # Write the summaries and print an overview fairly often.
                if step % 10 == 0:
                    # Print status to stdout.
                    logging.info('Step {0}: loss = {1:.2f} ({2:.3f} sec)'.format(step, loss_value, duration))
                    # Update the events file.

                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if (step + 1) % exp_config.train_eval_frequency == 0:

                    logging.info('Training Data Eval:')
                    [train_loss, train_dice] = do_eval(exp_config,
                                                    sess,
                                                    eval_loss,
                                                    x_placeholder,
                                                    y_placeholder,
                                                    training_pl,
                                                    images_train,
                                                    labels_train,
                                                    exp_config.batch_size,
                                                    subset)

                    train_summary_msg = sess.run(train_summary, feed_dict={train_error_: train_loss,
                                                                           train_dice_: train_dice})
                    summary_writer.add_summary(train_summary_msg, step)

                    loss_history.append(train_loss)
                    if len(loss_history) > 5:
                        loss_history.pop(0)
                        loss_gradient = (loss_history[-5] - loss_history[-1]) / 2

                    logging.info('loss gradient is currently {0}'.format(loss_gradient))

                    if exp_config.schedule_lr and loss_gradient < exp_config.schedule_gradient_threshold:
                        logging.warning('Reducing learning rate!')
                        curr_lr /= 10.0
                        logging.info('Learning rate changed to: {0}'.format(curr_lr))

                        # reset loss history to give the optimisation some time to start decreasing again
                        loss_gradient = np.inf
                        loss_history = []

                    if train_loss <= last_train:  # best_train:
                        logging.info('Decrease in training error!')
                    else:
                        logging.info('No improvment in training error for {0} steps'.format(no_improvement_counter))

                    last_train = train_loss

                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % exp_config.val_eval_frequency == 0:

                    checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step)
                    # Evaluate against the training set.

                    if not exp_config.train_on_all_data:

                        # Evaluate against the validation set.
                        logging.info('Validation Data Eval:')
                        [val_loss, val_dice] = do_eval(exp_config,
                                                       sess,
                                                       eval_loss,
                                                       x_placeholder,
                                                       y_placeholder,
                                                       training_pl,
                                                       images_val,
                                                       labels_val,
                                                       exp_config.batch_size,
                                                       subset)

                        val_summary_msg = sess.run(val_summary, feed_dict={val_error_: val_loss,
                                                                           val_dice_: val_dice})
                        summary_writer.add_summary(val_summary_msg, step)

                        if val_dice > best_dice:
                            best_dice = val_dice
                            best_file = os.path.join(log_dir, 'model_best_dice.ckpt')
                            saver_best_dice.save(sess, best_file, global_step=step)
                            logging.info('Found new best dice on validation set! - %f -  Saving model_best_dice.ckpt' % val_dice)

                        if val_loss < best_val:
                            best_val = val_loss
                            best_file = os.path.join(log_dir, 'model_best_xent.ckpt')
                            saver_best_xent.save(sess, best_file, global_step=step)
                            logging.info('Found new best crossentropy on validation set! - %f -  Saving model_best_xent.ckpt' % val_loss)


                step += 1

        sess.close()
    ######################################################################

    data.close()


def do_eval(exp_config,
            sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size,
            subset='0'):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param images: A numpy array or h5py dataset containing the images
    :param labels: A numpy array or h45py dataset containing the corresponding labels 
    :param batch_size: The batch_size to use. 
    :return: The average loss (as defined in the experiment), and the average dice over all `images`. 
    '''
    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in BackgroundGenerator(iterate_minibatches(exp_config, images, labels, batch_size=batch_size, augment_batch=False)):  # No aug in evaluation

        x, y = batch

        if y.shape[0] < batch_size:
            continue

        feed_dict = { images_placeholder: x,
                      labels_placeholder: y,
                      training_time_placeholder: False}

        closs, cdice, _ = sess.run(eval_loss, feed_dict=feed_dict)

        loss_ii += closs
        dice_ii += cdice

        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average loss: {0:.4f}, average accuracy: {1:.4f}'.format(avg_loss, avg_dice))

    return avg_loss, avg_dice


def augmentation_function(images, labels, exp_config, **kwargs):
    '''
    Function for augmentation of minibatches. It will transform a set of images and corresponding labels
    by a number of optional transformations. Each image/mask pair in the minibatch will be seperately transformed
    with random parameters. 
    :param images: A numpy array of shape [minibatch, X, Y, (Z), nchannels]
    :param labels: A numpy array containing a corresponding label mask
    :param do_rotations: Rotate the input images by a random angle between -15 and 15 degrees.
    :param do_scaleaug: Do scale augmentation by sampling one length of a square, then cropping and upsampling the image
                        back to the original size. 
    :param do_fliplr: Perform random flips with a 50% chance in the left right direction. 
    :return: A mini batch of the same size but with transformed images and masks. 
    '''

    do_rotations = kwargs.get('do_rotations', False)
    do_strong_rotations = kwargs.get('do_strong_rotations', False)
    do_shear = kwargs.get('do_shear', False)
    do_scaleaug = kwargs.get('do_scaleaug', False)
    do_fliplr = kwargs.get('do_fliplr', False)
    do_disturb = kwargs.get('do_disturb', False)
    mixup = kwargs.get('mixup', False)


    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for ii in range(num_images):

        img = np.squeeze(images[ii,...])
        lbl = np.squeeze(labels[ii,...])

        # ROTATE
        if do_rotations:
            angles = kwargs.get('angles', (-15,15))
            random_angle = np.random.uniform(angles[0], angles[1])
            img = image_utils.rotate_image(img, random_angle)
            lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

        # SHEAR
        if do_shear:
            shear_factor = kwargs.get('shear_factor', 0.2)
            img = image_utils.shear_image(img, shear_factor)
            lbl = image_utils.shear_image(lbl, shear_factor, interp=cv2.INTER_NEAREST)

        # STRONG ROTATE
        if do_strong_rotations:
            angles = kwargs.get('angles', (-90,90))
            random_angle = np.random.uniform(angles[0], angles[1])
            img = image_utils.rotate_image(img, random_angle)
            lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.flipud(img)
                lbl = np.flipud(lbl)

        # Add RANDOM NOISE to image
        if do_disturb > 0:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                noise_mask = np.random.normal(0, 0.1, size=img.shape)
                img = img + noise_mask

        # RANDOM CROP SCALE
        if do_scaleaug:
            offset = kwargs.get('offset', 30)
            n_x, n_y = img.shape
            r_y = np.random.random_integers(n_y-offset, n_y)
            p_x = np.random.random_integers(0, n_x-r_y)
            p_y = np.random.random_integers(0, n_y-r_y)

            img = image_utils.resize_image(img[p_y:(p_y+r_y), p_x:(p_x+r_y)],(n_x, n_y))
            lbl = image_utils.resize_image(lbl[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y), interp=cv2.INTER_NEAREST)

        # RANDOM FLIP
        if do_fliplr:
            coin_flip = np.random.randint(2)
            if coin_flip == 0:
                img = np.fliplr(img)
                lbl = np.fliplr(lbl)

        # MIXUP
        if mixup:
            if not (exp_config.batch_size % 2):
                raise ValueError('Batch size must be even to apply mixup. Current batch size is {}'.format(exp_config.batch_size))
            l = np.random.uniform()
            if ii % 2:
                img = l*np.squeeze(images[ii-1,...]) + (1-l)*img
            else:
                img = l*img + (1-l)*np.squeeze(images[ii+1,...])

        new_images.append(img[..., np.newaxis])
        new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    sampled_label_batch = np.asarray(new_labels)

    return sampled_image_batch, sampled_label_batch


def iterate_minibatches(exp_config, images, labels, batch_size, augment_batch=False):
    '''
    Function to create mini batches from the dataset of a certain batch size 
    :param images: hdf5 dataset
    :param labels: hdf5 dataset
    :param batch_size: batch size
    :param augment_batch: should batch be augmented?
    :return: mini batches
    '''

    random_indices = np.arange(images.shape[0])
    np.random.shuffle(random_indices)

    n_images = images.shape[0]

    for b_i in range(0, n_images, batch_size):

        if b_i + batch_size > n_images:
            continue

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        X = images[batch_indices, ...]
        y = labels[batch_indices, ...]

        image_tensor_shape = [X.shape[0]] + list(exp_config.image_size) + [1]
        X = np.reshape(X, image_tensor_shape)

        if augment_batch:
            X, y = augmentation_function(X, y, exp_config,
                                         do_rotations=exp_config.do_rotations,
                                         do_scaleaug=exp_config.do_scaleaug,
                                         do_fliplr=exp_config.do_fliplr,
                                         do_disturb=exp_config.do_disturb)

        yield X, y


def main(experiment, dataset, args):

    ### EXPERIMENT CONFIG FILE #############################################################
    # Set the config file of the experiment you want to run here:
    wd = os.path.dirname(os.path.realpath(__file__))

    sys.path.append(os.path.join(wd, 'cardiac-segmentation'))
    Model = importlib.import_module('cardiac-segmentation.model').Model

    exp_config = importlib.import_module('experiments.{0}'.format(experiment))


    # Modify settings with parameters passed as arguments
    exp_config.experiment_name = exp_config.experiment_name + args.name
    if args.batch_size > 1:
        exp_config.batch_size = args.batch_size
    if np.any((args.rotation, args.strong_rotation, args.shear, args.scale, args.flip, args.disturb)):
        exp_config.augment_batch = True
        exp_config.do_rotations = args.rotation
        exp_config.do_strong_rotations = args.strong_rotation
        exp_config.do_shear = args.shear
        exp_config.do_scaleaug = args.scale
        exp_config.do_fliplr = args.flip
        exp_config.do_disturb = 0.1*args.disturb
    
    if args.epochs is not None:
        exp_config.max_epochs = args.epochs

    ########################################################################################

    run_training(Model, exp_config, dataset, args.subset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train a neural network.")
    parser.add_argument("EXPERIMENT", type=str, help="Name of experiment to use.")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset to use for training.")
    parser.add_argument("--subset", type=str, default='training', help="Dataset subset to user for particular dataset chosen.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("-r", "--rotation", action='store_true', help="Rotation augmentations.")
    parser.add_argument("-sr", "--strong_rotation", action='store_true', help="Strong rotation augmentations.")
    parser.add_argument("--shear", action='store_true', help="Perform shear on image.")
    parser.add_argument("-s", "--scale", action='store_true', help="Scale augmentations.")
    parser.add_argument("-f", "--flip", action='store_true', help="Flip augmentations.")
    parser.add_argument("--disturb", type=int, default=0, help="Perform image disturbance.")
    parser.add_argument("-m", "--mixup", action='store_true', help="Perform image mixup.")
    parser.add_argument("-n", "--name", type=str, default='', help="Name suffix to add to the experiment.")
    parser.add_argument("-e", "--epochs", type=int, default=20000, help="Max number of epochs to compute.")
    args = parser.parse_args()
    all_experiments = [f[:-3] for f in os.listdir(os.path.join(sys_config.project_root, 'experiments')) if f[:2] != '__']
    assert args.EXPERIMENT in all_experiments, 'Experiment "{0}" not available. \
        Please, choose one of the following: {1}'.format(args.EXPERIMENT, all_experiments)

    dataset = args.dataset
    main(args.EXPERIMENT, dataset, args)
