import nibabel as nib


def load_nii(img_path):
    '''
    Shortcut to load a nifti file
    '''
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_nii(img_path, data, affine, header):
    '''
    Shortcut to save a nifty file
    '''
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)
