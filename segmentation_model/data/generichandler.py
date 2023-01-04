# -*- coding: utf-8 -*-
import os
import glob
from data.nifti import Nifti


class GenericHandler(object):
    '''
    Class for handling nifti datasets.
    '''
    def __init__(self, mode='', datapath=''):
        '''Constructor.'''
        self.dataset   = self.__class__.__name__.lower() # Name of class in lowercase format.
        wd = os.path.dirname(os.path.realpath(__file__))
        if len(datapath):
            wd = datapath
        self.base_path = os.path.join(wd, self.dataset)
        if len(mode):
            self.base_path = os.path.join(self.base_path, mode)

    def get_files(self):
        '''Obtain all file paths for each of the corresponding sets (training and testing).'''
        return self.get_files_in_folder(self.base_path)

    def get_files_in_folder(self, folder):
        '''
        Process all files inside given folder and extract available information.
        Returns:
            returns a list of files with keys info, image, mask, phase, patient_id
        '''
        file_list = []
        for f in glob.iglob(os.path.join(self.base_path, '*', '*_sa_??.nii.gz')):
            pat = os.path.basename(f).rstrip('.nii.gz').split('_')
            patient_id, _, phase = pat
            new_file = {'image': Nifti(f), 'mask': '', 'patient_id': patient_id, 'phase': phase, 'suffix': ''}

            maskf = f.rstrip('.nii.gz') + '_gt.nii.gz'
            if os.path.exists(maskf):
                new_file['mask'] = Nifti(maskf)

            file_list.append(new_file)

        return file_list