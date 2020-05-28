# -*- coding: utf-8 -*-
import os
import nibabel as nib


class Nifti(object):
    '''
    Nifti image class for loading all information.
    '''
    def __init__(self, path):
        '''Constructor.'''
        self.filetype = 'nifti'
        self.filepath = path
        nimg = nib.load(self.filepath)
        self.header = nimg.header
        self.affine = nimg.affine
        self.pixel_size = (self.header.structarr['pixdim'][1], 
                           self.header.structarr['pixdim'][2],
                           self.header.structarr['pixdim'][3])
        array = nimg.get_fdata()
        self.shape = array.shape
        array = None
        nimg = None

    def read(self):
        '''Read file content.'''
        nimg = nib.load(self.filepath)
        array = nimg.get_fdata()
        return array, self.shape, self.pixel_size
