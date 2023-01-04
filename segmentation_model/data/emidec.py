# -*- coding: utf-8 -*-
import os
import re
import glob

wd = os.path.dirname(os.path.realpath(__file__))

try:
    from data.generichandler import GenericHandler
    from data.nifti import Nifti
except ImportError:
    import sys
    sys.path.append('/'.join(wd.split('/')[:-1]))
    from data.generichandler import GenericHandler
    from data.nifti import Nifti


class Emidec(GenericHandler):
    '''
    Class for handling data files from EMIDEC dataset.
    '''
    def get_files_in_folder(self, folder):
        '''
        Process all files inside given folder and extract available information.
        Returns:
            returns a list of files with keys info, image, mask, phase, patient_id
        '''
        file_list = []
        for f in glob.iglob(os.path.join(self.base_path, 'patient*_EMIDEC', 'patient*_LGE.nii.gz')):
            pat = os.path.basename(f).rstrip('.nii.gz').split('_')
            patient_id, _ = pat
            new_file = {'image': Nifti(f), 'mask': '', 'patient_id': patient_id[7:], 'phase': '', 'suffix': ''}

            maskf = f.rstrip('.nii.gz') + '_gt.nii.gz'
            if os.path.exists(maskf):
                new_file['mask'] = Nifti(maskf)

            file_list.append(new_file)

        return file_list
