from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import importlib
import argparse
import glob
import sys
import cv2
import os

from skimage import transform

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = '/'.join(dir_path.split('/')[:-2])
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'cardiac-segmentation'))

import config.system as config
from data import dataset


def load_image(filepath):
    suff = os.path.basename(filepath)[-3:]
    if suff == 'png':
        return cv2.imread(filepath, 0)
    else:
        return nib.load(filepath).get_fdata()


def merge_image_prediction(img, pred):
    """
    Merge integer-valued image and label prediction in one RGB image.
    Parameters:
        img: Must be a 2D array with integer values.
        pred: Must be a 2D array with values 0,1,2,3 for BG,LV,MYO,RV, respect.
    Returns:
        A RGB image in the form of a 3d numpy array.
    """
    finalimg = np.zeros((*img.shape, 3), dtype=float)
    if finalimg.shape[:2] != pred.shape:
        aux = transform.resize(pred,
            (finalimg.shape[0], finalimg.shape[1], 1),
            order=1,
            preserve_range=True,
            mode='constant')
        pred_rsz = aux.reshape(aux.shape[:2])
    else:
        pred_rsz = pred
    # Red channel
    finalimg[...,0] = img[:]/np.max(img)
    finalimg[...,0][pred_rsz==1] = 1.
    # Green channel
    finalimg[...,1] = img[:]/np.max(img)
    finalimg[...,1][pred_rsz==2] = 1.
    # Blue channel
    finalimg[...,2] = img[:]/np.max(img)
    finalimg[...,2][pred_rsz==3] = 1.
    
    return finalimg


def plot_page(model_path, dataset, patientid, suffix, images, phases):
    """
    Plots all images in a pdf page.
    """
    summ_folder = os.path.join(model_path, dataset, 'summary')
    if not os.path.exists(summ_folder):
        os.makedirs(summ_folder)

    if len(images['orig'].keys()) == 2:
        ncols = max(images['orig']['ED'].shape[2], images['orig']['ES'].shape[2])
        nrows = 8 # 4 for each cardiac phase
    else:
        ncols = images['orig'][phases[0]].shape[2]
        nrows = 4

    f, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(2*ncols,2*nrows))
    f.suptitle('Patient {0} {1} \nDataset: {2} \nModel: {3}'.format(patientid, suffix, dataset[0].upper() + dataset[1:], model_path.split('/')[-1]), 
               fontsize=16)
    f.subplots_adjust(hspace=0.1, wspace=0.1)
    for i, ph in enumerate(phases):
        for c in range(ncols):
            # Prediction and ground truth/original
            if c < images['orig'][ph].shape[2]:
                axes[0+i*nrows//2,c].imshow(images['orig'][ph][...,c], cmap='gray')
                axes[1+i*nrows//2,c].imshow(merge_image_prediction(images['orig'][ph][...,c], images['gt'][ph][...,c]))
                axes[2+i*nrows//2,c].imshow(merge_image_prediction(images['orig'][ph][...,c], images['pred'][ph][...,c]))
                axes[3+i*nrows//2,c].imshow(images['diff'][ph][...,c], cmap='gray')
            # axes[0+i*nrows//2,c].axis('off'), axes[1+i*nrows//2,c].axis('off'), axes[2+i*nrows//2,c].axis('off'), axes[3+i*nrows//2,c].axis('off')
            axes[0+i*nrows//2,c].tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False)
            plt.setp(axes[0+i*nrows//2,c].get_yticklabels(), visible=False)
            axes[1+i*nrows//2,c].tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False)
            plt.setp(axes[1+i*nrows//2,c].get_yticklabels(), visible=False)
            axes[2+i*nrows//2,c].tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False)
            plt.setp(axes[2+i*nrows//2,c].get_yticklabels(), visible=False)
            axes[3+i*nrows//2,c].tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False)
            plt.setp(axes[3+i*nrows//2,c].get_yticklabels(), visible=False)
            
        axes[0+i*nrows//2,0].set_ylabel('{0}\nOriginal'.format(ph),     fontsize=10)
        axes[1+i*nrows//2,0].set_ylabel('{0}\nGround truth'.format(ph), fontsize=10)
        axes[2+i*nrows//2,0].set_ylabel('{0}\nPrediction'.format(ph),   fontsize=10)
        axes[3+i*nrows//2,0].set_ylabel('{0}\nDifference'.format(ph),   fontsize=10)

    f.savefig(os.path.join(summ_folder, 'patient{0}{1}.pdf'.format(patientid, suffix)))
    plt.close()


def generate_patient_sheet(volumes, model_path, subfolder, dataset):
    """
    Generates a sheet for ED and ES segmentation predictions and groundtruths
    for every slice in those phases.
    """
    print('Generating summary for patient {0} {1}.'.format(str(volumes['ID']).zfill(3), volumes['suffix'][1:]))
    # Training or testing patient
    pred_folder = os.path.join(model_path, dataset, subfolder, 'prediction')
    diff_folder = os.path.join(model_path, dataset, subfolder, 'difference')

    images = {'orig': {}, 'gt': {}, 'pred': {}, 'diff': {}}
    phases = ['ED', 'ES']

    avail_phases = []
    for ph in phases:
        if volumes[ph] is None:
            continue
        filename = os.path.basename(volumes[ph].filepath)
        avail_phases.append(ph)
        # Original
        images['orig'][ph] = volumes[ph].image.read()[0]
        # Ground truth and difference
        if volumes[ph].mask_exist:
            images['gt'][ph] = volumes[ph].mask.read()[0]
            images['diff'][ph] = load_image(os.path.join(diff_folder, filename) )
        else:
            images['gt'][ph] = np.zeros(images['orig'][ph].shape)
            images['diff'][ph] = np.zeros(images['orig'][ph].shape)
        # Prediction
        images['pred'][ph] = load_image(os.path.join(pred_folder, filename))

    plot_page(model_path, dataset, volumes['ID'], volumes['suffix'][1:], images, avail_phases)


def summarize_dataset(dataset_name, exp_config, model_path):
    '''Iterate over evaluated patients.'''
    dataset_folder = os.path.join(model_path, dataset_name)
    pred_dict = {'predictions': 'training', 'predictions_testset': 'testing'}
    for f in glob.iglob(os.path.join(dataset_folder, 'pred*')):
        # if f.split('/')[-2] != 'mscmrseg':
        #     continue
        subfolder = f.split('/')[-1]
        data = dataset.Dataset(dataset_name, pred_dict[subfolder], 
                               exp_config.data_mode, 
                               exp_config.image_size, 
                               exp_config.target_resolution)
        vol_dict = {'ED': None, 'ES': None, 'ID': '', 'suffix': ''}
        for volume in data:
            if vol_dict['ID'] == '':
                # Create new patient dict.
                vol_dict['ID'] = volume.patient_id
                vol_dict['suffix'] = volume.suffix
                vol_dict[volume.phase] = volume
            elif volume.patient_id == vol_dict['ID'] and volume.suffix == vol_dict['suffix']:
                # Add the other phase to patient dict.
                vol_dict[volume.phase] = volume
            else:
                # New patient. Generate sheet and restart patient dict.
                generate_patient_sheet(vol_dict, model_path, subfolder, dataset_name)
                vol_dict = {'ED': None, 'ES': None, 'ID': '', 'suffix': ''}
                vol_dict['ID'] = volume.patient_id
                vol_dict['suffix'] = volume.suffix
                vol_dict[volume.phase] = volume

            # Do plot for last volumes
            if data.current > data.high:
                generate_patient_sheet(vol_dict, model_path, subfolder, dataset_name)




def summary(model, data=None):
    """Computes summary for predicted images."""
    importlib.reload(plt)
    model_path = os.path.join(config.log_root, model)
    experiment = list(glob.iglob(os.path.join(model_path, '*.py')))[0]
    exp_config = importlib.machinery.SourceFileLoader(experiment.split('/')[-1].rstrip('.py'), experiment).load_module()
    if data is not None:
        summarize_dataset(data, exp_config, model_path)
    else:
        # Find all folders of evaluated datasets for given model
        for f in os.listdir(model_path):
            data = f
            path = os.path.join(model_path, data)
            if os.path.isdir(path) and data[:2] != '__':
                print('Dataset', data)
                summarize_dataset(data, exp_config, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot segmentation predictions and compare them with \
        background truth or original image."
    )

    model_folders = [n for n in os.listdir(config.log_root) \
                     if os.path.isdir(os.path.join(config.log_root, n))]
    parser.add_argument('-m', '--model', type=str, 
        help="Name of model to use for predictions. Possible models: {0}".format(model_folders)
    )

    args = parser.parse_args()
    assert args.model in model_folders, 'Selected model ({0}) is not available. \
        Please choose one of the following: {1}'.format(args.model, model_folders)

    summary(args.model)
