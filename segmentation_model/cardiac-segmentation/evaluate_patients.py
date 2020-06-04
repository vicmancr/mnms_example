import os
import re
import cv2
import glob
import time
import shutil
import logging
import numpy as np
import tensorflow as tf

from skimage import transform

from utils import utils_gen, utils_nii, image_utils
from data.dataset import Dataset



def crop_or_pad_volume_to_size(vol, nx, ny):

    x, y, z = vol.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        vol_cropped = vol[x_s:x_s + nx, y_s:y_s + ny, :]
    else:
        vol_cropped = np.zeros((nx, ny, z))
        if x <= nx and y > ny:
            vol_cropped[x_c:x_c + x, :, :] = vol[:, y_s:y_s + ny, :]
        elif x > nx and y <= ny:
            vol_cropped[:, y_c:y_c + y, :] = vol[x_s:x_s + nx, :, :]
        else:
            vol_cropped[x_c:x_c + x, y_c:y_c + y, :] = vol[:, :, :]

    return vol_cropped


def score_data(model, output_folder, input_datapath, model_path, dataset, exp_config, 
               do_postprocessing=False):

    batch_size = 1
    num_channels = exp_config.nlabels

    image_tensor_shape = [batch_size] + list(exp_config.image_size) + [1]
    images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')

    mask_pl, softmax_pl = model.predict(images_pl)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # Backward compatibility
    if 'model_type' not in exp_config.__dict__.keys():
        exp_config.model_type = 'convolutional'

    with tf.Session() as sess:

        sess.run(init)
        checkpoint_path = utils_gen.get_latest_model_checkpoint_path(model_path, 'model_best_dice.ckpt')

        saver.restore(sess, checkpoint_path)

        init_iteration = int(checkpoint_path.split('/')[-1].split('-')[-1])

        total_time = 0
        total_volumes = 0

        # Select image pixel size
        nx, ny = exp_config.image_size[:2]

        data = Dataset(dataset, '', exp_config.data_mode, input_datapath, (nx, ny), exp_config.target_resolution)
        gt_exists = False

        # Iterate over volumes and slices of dataset
        for volume in data:
            gt_exists = volume.mask_exist
            predictions = []

            logging.info(' ----- Doing image: -------------------------')
            logging.info('  Doing: {}'.format(volume.filepath))
            logging.info(' --------------------------------------------')

            start_time = time.time()

            for slc in volume:

                x, y = slc.shape
                img = slc.img
                slice_cropped = slc.img_cropped
                x_s, y_s, x_c, y_c = slc.cropped_boundaries
                if gt_exists:
                    mask = slc.mask

                # GET PREDICTION
                network_input = np.float32(np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (batch_size, 1, 1, 1)))
                _, logits_out = sess.run([mask_pl, softmax_pl], feed_dict={images_pl: network_input})
                prediction_cropped = np.squeeze(logits_out[0,...])

                # ASSEMBLE BACK THE SLICES
                slice_predictions = np.zeros((x,y,num_channels))
                # insert cropped region into original image again
                if x > nx and y > ny:
                    slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = prediction_cropped
                else:
                    if x <= nx and y > ny:
                        slice_predictions[:, y_s:y_s+ny,:] = prediction_cropped[x_c:x_c+ x, :,:]
                    elif x > nx and y <= ny:
                        slice_predictions[x_s:x_s + nx, :,:] = prediction_cropped[:, y_c:y_c + y,:]
                    else:
                        slice_predictions[:, :,:] = prediction_cropped[x_c:x_c+ x, y_c:y_c + y,:]

                # RESCALING ON THE LOGITS
                if gt_exists:
                    prediction = transform.resize(slice_predictions,
                                                  (*mask.shape[:2], num_channels),
                                                  order=1,
                                                  preserve_range=True,
                                                  mode='constant')
                else:  # This can occasionally lead to wrong volume size, therefore if gt_exists
                        # we use the gt mask size for resizing.
                    prediction = transform.resize(slice_predictions,
                                                  (*img.shape[:2], num_channels),
                                                  order=1,
                                                  preserve_range=True,
                                                  mode='constant')


                prediction = np.uint8(np.argmax(prediction, axis=-1))
                predictions.append(prediction)

            prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))

            if do_postprocessing:
                prediction_arr = image_utils.keep_largest_connected_components(prediction_arr)

            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            total_volumes += 1

            logging.info('Evaluation of volume took {0} secs.'.format(elapsed_time))

            out_basepath = os.path.join(output_folder, 'prediction')
            in_basepath = os.path.join(output_folder, 'image')

            filebasename = os.path.basename(volume.filepath)
            out_file_name = os.path.join(out_basepath, filebasename)
            image_file_name = os.path.join(in_basepath, filebasename)

            if volume.image.filetype == 'nifti':
                # Save prediced mask
                out_affine, out_header = volume.image.affine, volume.image.header
                utils_nii.save_nii(out_file_name, prediction_arr, out_affine, out_header)
                # Save image data in model folder for convenience
                utils_nii.save_nii(image_file_name, volume.image.read()[0], out_affine, out_header)
            else:
                # Save prediced mask
                cv2.imwrite(out_file_name, prediction_arr)
                # Save image data in model folder for convenience
                cv2.imwrite(image_file_name, volume.image.read()[0])

            logging.info('saved image to: {}'.format(image_file_name))
            logging.info('saved prediction to: {}'.format(out_file_name))

            if gt_exists:

                # Save GT image
                gt_filebase = os.path.join(output_folder, 'ground_truth')
                logging.info('saving to: {}'.format(gt_filebase))
                shutil.copy(volume.maskpath, os.path.join(gt_filebase, filebasename))

                # Save difference mask between predictions and ground truth
                mask = mask.reshape((*mask.shape, 1))
                difference_mask = np.where(np.abs(prediction_arr - mask) > 0, [1], [0])
                difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                diff_filebase = os.path.join(output_folder, 'difference')
                diff_file_name = os.path.join(diff_filebase, filebasename)
                if volume.image.filetype == 'nifti':
                    utils_nii.save_nii(diff_file_name, difference_mask, out_affine, out_header)
                else:
                    cv2.imwrite(diff_file_name, difference_mask)

                logging.info('saved to: {}'.format(diff_file_name))


        logging.info('Average time per volume: {}'.format(total_time/total_volumes))

    return init_iteration
