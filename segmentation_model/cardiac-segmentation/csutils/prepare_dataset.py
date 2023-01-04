# -*- coding: utf-8 -*-
import numpy as np
import argparse
import logging
import h5py
import glob
import sys
import gc
import re
import os

from skimage import transform
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append('/'.join(dir_path.split('/')[:-2]))

import config.system as config

from utils import utils_gen, utils_nii, image_utils
from data.dataset import Dataset

parser = argparse.ArgumentParser(
    description="Prepare dataset and save in hdf5 format for training.")

data_folders = [n for n in os.listdir(config.data_base) \
                if os.path.isdir(os.path.join(config.data_base, n))]
parser.add_argument('-d', '--data', type=str, 
    help="Name of dataset to use. Possible datasets: {0}".format(data_folders))

parser.add_argument("--subset", type=str, default='training', 
    help="Dataset subset to use for particular dataset chosen.")

avail_modes = ['2D']
parser.add_argument('-m', '--mode', type=str, default='2D',
    help="Mode of dataset to use. Possible modes: {0}".format(avail_modes))

avail_model_types = ['convolutional']
parser.add_argument('-t', '--model_type', type=str, default='convolutional',
    help="Model type related to dataset. Possible model types: {0}".format(avail_model_types))

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5


class DataManager(object):
    def __init__(self, filepath, data, size, model_type, split_test_train):
        self.dataset = data
        self.hdf5_file = h5py.File(filepath, "w")
        self.model_type = model_type
        self.split_test_train = split_test_train
        self.write_buffer = 0
        self.counter_from = 0
        
        self.cardiac_phase_list = {'test': [], 'train': []}

        self.mask_list = {'test': [], 'train': [] }
        self.img_list = {'test': [], 'train': [] }

        self.count_slices()
        self.create_datasets()
        self.create_image_datasets(size)

    def _release_tmp_memory(self, train_or_test):
        '''Helper function to reset the tmp lists and free the memory.'''
        self.img_list[train_or_test].clear()
        self.mask_list[train_or_test].clear()
        gc.collect()

    def _write_range_to_hdf5(self, train_or_test, counter_from, counter_to):
        '''Helper function to write a range of data to the hdf5 datasets.'''

        logging.info('Writing data from {0} to {1}'.format(counter_from, counter_to))

        img_arr = np.asarray(self.img_list[train_or_test], dtype=np.float32)
        mask_arr = np.asarray(self.mask_list[train_or_test], dtype=np.uint8)

        self.hdf5_data['data_{}'.format(train_or_test)][counter_from:counter_to, ...] = img_arr
        self.hdf5_data['pred_{}'.format(train_or_test)][counter_from:counter_to, ...] = mask_arr

    def count_slices(self):
        '''Count number of slices that the final dataset will have.'''
        self.file_list = {'test': [], 'train': []}
        self.num_slices = {'test': 0, 'train': 0}

        for volume in self.dataset:
            if self.split_test_train:
                if int(volume.patient_id) > 68:
                    train_test = 'test'
                else:
                    train_test = 'train'
            else:
                train_test = 'train'

            self.file_list[train_test].append(volume)

            frame = volume.phase
            if frame == 'ES':
                self.cardiac_phase_list[train_test].append(1)  # 1 == systole
            elif frame == 'ED':
                self.cardiac_phase_list[train_test].append(2)  # 2 == diastole
            else:
                self.cardiac_phase_list[train_test].append(0)  # 0 means other phase

            self.num_slices[train_test] += volume.shape[2]

            if train_test == 'test':
                print('Volume shape', volume.shape[2])
                print('Total shape', self.num_slices[train_test])


    def create_datasets(self):
        # Write the small datasets
        for tt in ['test', 'train']:
            self.hdf5_file.create_dataset('cardiac_phase_{}'.format(tt), data=np.asarray(self.cardiac_phase_list[tt], dtype=np.uint8))

    def create_image_datasets(self, size):
        '''Create datasets to save the images in array format.'''
        self.nx, self.ny = size
        n_train = self.num_slices['train']
        n_test = self.num_slices['test']

        # Create datasets for images and masks
        self.hdf5_data = {}
        for tt, num_points in zip(['test', 'train'], [n_test, n_train]):
            if num_points > 0:
                self.hdf5_data['data_{}'.format(tt)] = self.hdf5_file.create_dataset('data_{}'.format(tt), [num_points] + list(size), dtype=np.float32)
                self.hdf5_data['pred_{}'.format(tt)] = self.hdf5_file.create_dataset('pred_{}'.format(tt), [num_points] + list(size), dtype=np.uint8)

    def start(self):
        self.write_buffer = 0
        self.counter_from = 0

    def add(self, train_or_test, images):
        '''Add images to dataset lists.'''
        self.img_list[train_or_test].append(images[0])
        self.mask_list[train_or_test].append(images[1])

        self.write_buffer += 1

        if self.write_buffer >= MAX_WRITE_BUFFER:

            counter_to = self.counter_from + self.write_buffer
            self._write_range_to_hdf5(train_or_test, self.counter_from, counter_to)
            self._release_tmp_memory(train_or_test)

            # reset stuff for next iteration
            self.counter_from = counter_to
            self.write_buffer = 0

    def end(self, train_or_test):
        '''Save remaning data for given subset.'''
        logging.info('Writing remaining data')
        counter_to = self.counter_from + self.write_buffer

        self._write_range_to_hdf5(train_or_test, self.counter_from, counter_to)
        self._release_tmp_memory(train_or_test)

    def close(self):
        self.hdf5_file.close()



def crop_or_pad_slice_to_size(slice, nx, ny):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def prepare_data(dataset, output_file, subset, mode, model_type, size, target_resolution, split_test_train=True):
    '''
    Main function that prepares a dataset from the raw ACDC data to an hdf5 dataset
    '''

    assert (mode in avail_modes), 'Unknown mode: {0}'.format(mode)
    assert (model_type in avail_model_types), 'Unknown model type: {0}'.format(model_type)

    logging.info('Counting files and parsing meta data...')

    data = Dataset(dataset, subset, mode, '', size, target_resolution)
    data_manager = DataManager(output_file, data, size, model_type, split_test_train)

    img_mask_processing = {
        '2D_convolutional': img_mask_2D
    }

    logging.info('Parsing image files')
    img_mask_processing['_'.join([mode, model_type])](data_manager)



def img_mask_2D(data_manager):
    
    train_test_range = ['test', 'train'] if data_manager.split_test_train else ['train']
    for train_test in train_test_range:

        data_manager.start()
        for vol in data_manager.file_list[train_test]:
            logging.info('-'*20)
            logging.info('Doing: {}'.format(vol.filepath))

            img = vol.image.read()[0]
            mask = vol.mask.read()[0]
            pixel_size = vol.pixel_size

            img = image_utils.normalise_image(img, 0, 1)

            ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################

            scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]

            for zz in range(img.shape[2]):
                slice_img = np.squeeze(img[:, :, zz])
                slice_rescaled = transform.rescale(slice_img,
                                                   scale_vector,
                                                   order=1,
                                                   preserve_range=True,
                                                   multichannel=False,
                                                   mode = 'constant')

                slice_mask = np.squeeze(mask[:, :, zz])
                mask_rescaled = transform.rescale(slice_mask,
                                                  scale_vector,
                                                  order=0,
                                                  preserve_range=True,
                                                  multichannel=False,
                                                  anti_aliasing=False,
                                                  mode='constant')

                slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, data_manager.nx, data_manager.ny)
                mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, data_manager.nx, data_manager.ny)

                data_manager.add(train_test, (slice_cropped, mask_cropped))

        data_manager.end(train_test)

    data_manager.close()




if __name__ == "__main__":

    args = parser.parse_args()
    dataset = args.data
    mode = args.mode
    subset = '_' + args.subset if args.subset != '' else args.subset
    model_type = args.model_type

    if '-' in dataset:
        datasets = dataset.split('-')
        logging.info('Evaluating more than one dataset at the same time: {0}'.format(datasets))
        for ds in datasets:
            assert ds in data_folders, 'Dataset "{0}" not available. \
                Please, choose one of the following: {1}'.format(ds, data_folders)
    else:
        assert dataset in data_folders, 'Dataset "{0}" not available. \
                Please, choose one of the following: {1}'.format(dataset, data_folders)

    preprocessing_folder = os.path.join(config.data_base, args.data, 'preproc_data')
    if not os.path.exists(preprocessing_folder):
        os.makedirs(preprocessing_folder)

    split_test_train = True
    subset_type = 'training'
    size = (256, 256)
    target_resolution = (1, 1)
    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])
    if split_test_train:
        data_file_name = 'data_{0}_{1}{2}_size_{3}_res_{4}_onlytrain.hdf5'.format(dataset, mode, subset, size_str, res_str)
    else:
        data_file_name = 'data_{0}_{1}{2}_size_{3}_res_{4}.hdf5'.format(dataset, mode, subset, size_str, res_str)
    output_file = os.path.join(preprocessing_folder, data_file_name)

    database = dataset if '-' not in dataset else datasets
    prepare_data(database, output_file, subset_type, mode, model_type, size, target_resolution, split_test_train)
