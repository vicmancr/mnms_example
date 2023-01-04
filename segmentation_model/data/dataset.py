import os
import re
import glob
import importlib
import numpy as np
import nibabel as nib

from skimage import transform
from PIL import Image

from utils import image_utils


class Slice(object):
    '''
    Class for a 2D slice.
    '''
    def __init__(self, image, mask, scale_vector, target_size=None):
        '''
        Constructor. It accounts for image, mask and properties.
        Parameters:
            image: Image data.
            mask: Mask data.
        '''
        self.img          = image
        self.mask         = mask
        self.scale_vector = scale_vector
        if target_size is not None:
            self.target_size  = target_size
        else:
            self.target_size = self.img.shape
        self.rescale_and_crop()

    def rescale_and_crop(self):
        '''Rescale and crop 2D slice.'''
        slice_img = np.squeeze(self.img)
        slice_rescaled = transform.rescale(slice_img, self.scale_vector, order=1, multichannel=False,
                                           preserve_range=True, mode='constant')
        slice_mask = np.squeeze(self.mask)
        mask_rescaled = np.round(transform.rescale(slice_mask, self.scale_vector, order=0, multichannel=False,
                                           preserve_range=True, mode='constant'))

        self.shape = slice_rescaled.shape
        x, y = self.shape
        nx, ny = self.target_size[:2]

        x_s = (x - nx) // 2
        y_s = (y - ny) // 2
        x_c = (nx - x) // 2
        y_c = (ny - y) // 2
        self.cropped_boundaries = (x_s, y_s, x_c, y_c)

        # Crop section of image for prediction
        if x > nx and y > ny:
            slice_cropped = slice_rescaled[x_s:x_s+nx, y_s:y_s+ny]
            mask_cropped = mask_rescaled[x_s:x_s+nx, y_s:y_s+ny]
        else:
            slice_cropped = np.zeros((nx,ny))
            mask_cropped = np.zeros((nx,ny))
            if x <= nx and y > ny:
                slice_cropped[x_c:x_c+ x, :] = slice_rescaled[:,y_s:y_s + ny]
                mask_cropped[x_c:x_c+ x, :] = mask_rescaled[:,y_s:y_s + ny]
            elif x > nx and y <= ny:
                slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
                mask_cropped[:, y_c:y_c + y] = mask_rescaled[x_s:x_s + nx, :]
            else:
                slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]
                mask_cropped[x_c:x_c+x, y_c:y_c + y] = mask_rescaled[:, :]

        self.img_cropped = slice_cropped
        self.mask_cropped = mask_cropped


class Volume(object):
    '''
    Class for a single volume.
    '''
    def __init__(self, fileinfo, image_size=None, image_resolution=None):
        '''
        Constructor. It defines the configuration for the volume.
        Parameters:
            filepath: Path to volume.
            maskpath: Path to mask if it exists.
            image_size: Desired final image size for cropping.
                If None, original size is preserved.
            image_resolution: Desired final image resolution.
                If None, image is not rescaled.
        '''
        self.image      = fileinfo['image']
        self.filepath   = fileinfo['image'].filepath
        self.mask_exist = False
        self.maskpath = ''
        if fileinfo['mask'] != '':
            self.maskpath   = fileinfo['mask'].filepath
            self.mask_exist = True
            self.mask = fileinfo['mask']
        self.patient_id = fileinfo['patient_id'] 
        self.info       = fileinfo['info'] if 'info' in fileinfo.keys() else {}
        self.phase      = fileinfo['phase'] if 'phase' in fileinfo.keys() else ''
        self.suffix     = fileinfo['suffix'] if 'suffix' in fileinfo.keys() else ''
        self.frame_suffix = '_' + self.phase + '_' + self.suffix
        self.image_size = image_size
        self.image_resolution = image_resolution
        self.slices  = []
        self.slices_index = {}
        self.current = 0
        self.shape = self.image.shape
        self.pixel_size = self.image.pixel_size

    def __iter__(self):
        return self

    def __next__(self):
        '''Returns next file.'''
        if self.current == 0: # Initialize when called
            self.process_slices()
            self.high = len(self.slices) - 1
        if self.current > self.high:
            self.slices = [] # Clear files
            self.img_dat = []
            self.mask_dat = []
            raise StopIteration
        else:
            self.current += 1
            return self.slices[self.current-1]

    def process_slices(self):
        '''Process slices of 3D images.'''
        img, _, _ = self.image.read()
        img = image_utils.normalise_image(img, 0, 1)

        if self.maskpath != '':
            mask = self.mask.read()[0]
        else:
            mask = np.zeros(img.shape)

        if self.image_resolution is not None:
            scale_vector = (self.pixel_size[0] / self.image_resolution[0], self.pixel_size[1] / self.image_resolution[1])
        else:
            scale_vector = (1, 1)
        self.rescale_and_crop(img, mask, scale_vector)

    def rescale_and_crop(self, img, mask, scale_vector):
        '''Rescale and crop slices of a 3D image.'''
        for zz in range(img.shape[2]):
            new_slc = Slice(img[:,:,zz], mask[:,:,zz], scale_vector, self.image_size)
            self.slices.append(new_slc)
            self.slices_index[self.filepath.split('/')[-1].split('.')[0]] = len(self.slices)-1

    def get_slice(self, idx):
        return self.slices[idx]


class Dataset(object):
    '''
    Class for handling data files for a given dataset.
    '''
    def __init__(self, dataset_name, subset='', mode='2D', datapath='', image_size=None, image_resolution=None):
        '''
        Constructor. It defines the configuration for the dataset handler.
        Parameters:
            data_base: Path to datasets.
            dataset_name: The dataset name to consider.
            subset: Set to return: train or test.
            mode: Type of data: 2D or 3D.
            image_size: Desired final image size for cropping.
            image_resolution: Desired final image resolution.
        '''
        self.mode = mode
        self.image_size = image_size
        self.image_resolution = image_resolution
        module = importlib.import_module('data.{0}'.format(dataset_name))
        handler = getattr(module, dataset_name[0].upper() + dataset_name[1:])
        generator = handler(subset, datapath)
        # File dictionaries containing data images.
        self.files = generator.get_files()
        self.volumes = []
        self.volumes_index = {}
        self.process_volumes()
        self.high    = len(self.volumes) - 1
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        '''Returns next file.'''
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.volumes[self.current-1]

    def process_volumes(self):
        '''Process slices of 3D images.'''
        for f in self.files:
            self.volumes.append(Volume(f, self.image_size, self.image_resolution))
            self.volumes_index[f['image'].filepath.split('/')[-1].split('.')[0]] = len(self.volumes)-1

    def get_volume_by_name(self, name):
        vol = self.volumes[self.volumes_index[name]]
        vol.current = 0
        vol.__next__()
        return vol
