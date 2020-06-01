"""
author: Clément Zotti (clement.zotti@usherbrooke.ca)
date: April 2017

DESCRIPTION :
The script provide helpers functions to handle nifti image format:
    - load_nii()
    - save_nii()

to generate metrics for two images:
    - metrics()

And it is callable from the command line (see below).
Each function provided in this script has comments to understand
how they works.

HOW-TO:

This script was tested for python 3.4.

First, you need to install the required packages with
    pip install -r requirements.txt

After the installation, you have two ways of running this script:
    1) python metrics.py ground_truth/patient001_ED.nii.gz prediction/patient001_ED.nii.gz
    2) python metrics.py ground_truth/ prediction/

The first option will print in the console the dice and volume of each class for the given image.
The second option wiil ouput a csv file where each images will have the dice and volume of each class.


Link: http://acdc.creatis.insa-lyon.fr

"""

import os
import re
import cv2
import glob
import time
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from medpy.metric.binary import hd, dc

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns


HEADER = ["Name", "Dice LV", "Mod Dice LV", "Volume LV", "Err LV(ml)",
          "Dice RV", "Mod Dice RV", "Volume RV", "Err RV(ml)",
          "Dice MYO", "Mod Dice MYO", "Volume MYO", "Err MYO(ml)"]

#
# Utils functions used to sort strings into a natural order
#
def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    """
    Sort a (list,tuple) of strings into natural order.

    Ex:

    ['1','10','2'] -> ['1','2','10']

    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']

    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


# code
def mod_dc(result, reference):
    r"""
    Dice coefficient
    
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.
    
    The metric is defined as
    
    .. math::
        
        DC=\frac{2|A\cap B|}{|A|+|B|}
        
    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).
        
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    
    intersection = np.count_nonzero(result & reference)
    
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 1.0
        
    return dc


#
# Utils function to load and save nifti files with the nibabel package
#
def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def load_png(img_path):
    array = cv2.imread(img_path)[...,0]
    array = array.reshape((*array.shape, 1))
    return array


def save_nii(img_path, data, affine, header):
    """
    Function to save a 'nii' or 'nii.gz' file.

    Parameters
    ----------

    img_path: string
    Path to save the image should be ending with '.nii' or '.nii.gz'.

    data: np.array
    Numpy array of the image data.

    affine: list of list or np.array
    The affine transformation to save with the image.

    header: nib.Nifti1Header
    The header that define everything about the data
    (pleasecheck nibabel documentation).
    """
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


#
# Functions to process files, directories and metrics
#
def metrics(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [3, 1, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Modified dice accuracy
        xent = mod_dc(gt_c_i, pred_c_i)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        # Compute volume
        volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice, xent, volpred, volpred-volgt]

    return res


#
# Functions to process files, directories and metrics
#
def metrics_slice(img_gt, img_pred, voxel_size):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res_zs = []
    for zz in range(img_gt.shape[2]):
        res_z = []
        # Loop on each classes of the input images
        for c in [3, 1, 2]:
            # Copy the gt image to not alterate the input
            gt_c_i = np.copy(img_gt[..., zz])
            gt_c_i[gt_c_i != c] = 0

            # Copy the pred image to not alterate the input
            pred_c_i = np.copy(img_pred[..., zz])
            pred_c_i[pred_c_i != c] = 0

            # Clip the value to compute the volumes
            gt_c_i = np.clip(gt_c_i, 0, 1)
            pred_c_i = np.clip(pred_c_i, 0, 1)

            # Modified dice accuracy
            xent = mod_dc(gt_c_i, pred_c_i)

            # Compute the Dice
            dice = dc(gt_c_i, pred_c_i)

            # Compute volume
            volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
            volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

            res_z += [dice, xent, volpred, volpred - volgt]

        res_zs += [res_z]

    return metrics(img_gt, img_pred, voxel_size), res_zs


def compute_metrics_on_files(path_gt, path_pred):
    """
    Function to give the metrics for two files

    Parameters
    ----------

    path_gt: string
    Path of the ground truth image.

    path_pred: string
    Path of the predicted image.
    """
    filetype = 'nifti'
    fileformat = os.path.basename(path_gt).split('.')[-1]
    if fileformat == 'png':
        filetype = 'image'
    
    if filetype == 'nifti':
        gt, _, header = load_nii(path_gt)
        pred, _, _ = load_nii(path_pred)
        zooms = header.get_zooms()
    else:
        gt = load_png(path_gt)
        pred = load_png(path_pred)
        zooms = (1,1,1)

    name = os.path.basename(path_gt)
    name = name.split('.')[0]
    res = metrics(gt, pred, zooms)
    res = ["{:.3f}".format(r) for r in res]

    formatting = "{:>14}, {:>7}, {:>9}, {:>10}, {:>7}, {:>9}, {:>10}, {:>8}, {:>10}, {:>11}"
    print(formatting.format(*HEADER))
    print(formatting.format(name, *res))


def compute_metrics_on_directories(dir_gt, dir_pred):
    """
    Function to generate a csv file for each images of two directories.

    Parameters
    ----------

    path_gt: string
    Directory of the ground truth segmentation maps.

    path_pred: string
    Directory of the predicted segmentation maps.
    """
    filetype = 'nifti'
    lst_gt = sorted(glob.glob(os.path.join(dir_gt, '*.nii.gz')), key=natural_order)
    lst_pred = sorted(glob.glob(os.path.join(dir_pred, '*.nii.gz')), key=natural_order)
    if len(lst_gt) == 0:
        filetype = 'image'
        lst_gt = sorted(glob.glob(os.path.join(dir_gt, '*.png')), key=natural_order)
        lst_pred = sorted(glob.glob(os.path.join(dir_pred, '*.png')), key=natural_order)

    res = []
    res_zs = []
    res_val = []
    res_zs_val = []
    if len(lst_pred) != len(lst_gt):
        gt_names = [os.path.basename(lg) for lg in lst_gt]
        lst_pred = list(filter(lambda x: os.path.basename(x) in gt_names, lst_pred))
    for p_gt, p_pred in zip(lst_gt, lst_pred):
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            continue

        if filetype == 'nifti':
            gt, _, header = load_nii(p_gt)
            pred, _, _ = load_nii(p_pred)
            zooms = header.get_zooms()
        else:
            gt = load_png(p_gt)
            pred = load_png(p_pred)
            zooms = (1,1,1)
        res_n, res_zsn = metrics_slice(gt, pred, zooms)
        res_zs.append(res_zsn)
        res.append(res_n)
        if int(os.path.basename(p_pred).split('_')[0][-1]) == 0:
            res_val.append(res_n)
            res_zs_val.append(res_zsn)

    lst_name_gt = [os.path.basename(gt).split(".")[0] for gt in lst_gt]
    new_res_zs = []
    for j, rz in enumerate(res_zs):
        new_res_zs += [ ['{}_z{}'.format(lst_name_gt[j], idx),] + r for idx, r in enumerate(rz)]
    dfd = pd.DataFrame(new_res_zs, columns=HEADER)
    dfd.to_csv(os.path.join(dir_pred, "results_detailed_{}.csv".format(time.strftime("%Y%m%d_%H%M%S"))), index=False)

    lst_name_gt = [os.path.basename(gt).split(".")[0] for gt in lst_gt]
    res = [[n,] + r for r, n in zip(res, lst_name_gt)]
    df = pd.DataFrame(res, columns=HEADER)
    df.to_csv(os.path.join(dir_pred, "results_{}.csv".format(time.strftime("%Y%m%d_%H%M%S"))), index=False)

    header_summary = ["Group", "Dice LV", "Std LV", "Mod Dice LV", "X Std LV", "Volume LV", "Std Vol LV", "Cum Err LV(ml)",
                      "Dice RV", "Std RV", "Mod Dice RV", "X Std RV", "Volume RV", "Std Vol RV", "Cum Err RV(ml)",
                      "Dice MYO", "Std MYO", "Mod Dice MYO", "X Std MYO", "Volume MYO", "Std Vol MYO", "Cum Err MYO(ml)"]
    summary  = [['Total', 
                df.iloc[:,1].mean(), df.iloc[:,1].std(), df.iloc[:,2].mean(), df.iloc[:,2].std(), df.iloc[:,3].mean(), df.iloc[:,3].std(), df.iloc[:,4].sum(),
                df.iloc[:,5].mean(), df.iloc[:,5].std(), df.iloc[:,6].mean(), df.iloc[:,6].std(), df.iloc[:,7].mean(), df.iloc[:,7].std(), df.iloc[:,8].sum(),
                df.iloc[:,9].mean(), df.iloc[:,9].std(), df.iloc[:,10].mean(), df.iloc[:,10].std(), df.iloc[:,11].mean(), df.iloc[:,11].std(), df.iloc[:,12].sum()
                ]]
    dfed = df[df['Name'].str.contains('_ED')]
    summary.append(['ED', 
                dfed.iloc[:,1].mean(), dfed.iloc[:,1].std(), dfed.iloc[:,2].mean(), dfed.iloc[:,2].std(), dfed.iloc[:,3].mean(), dfed.iloc[:,3].std(), dfed.iloc[:,4].sum(),
                dfed.iloc[:,5].mean(), dfed.iloc[:,5].std(), dfed.iloc[:,6].mean(), dfed.iloc[:,6].std(), dfed.iloc[:,7].mean(), dfed.iloc[:,7].std(), dfed.iloc[:,8].sum(),
                dfed.iloc[:,9].mean(), dfed.iloc[:,9].std(), dfed.iloc[:,10].mean(), dfed.iloc[:,10].std(), dfed.iloc[:,11].mean(), dfed.iloc[:,11].std(), dfed.iloc[:,12].sum()
                ])
    dfes = df[df['Name'].str.contains('_ES')]
    summary.append(['ES', 
                dfes.iloc[:,1].mean(), dfes.iloc[:,1].std(), dfes.iloc[:,2].mean(), dfes.iloc[:,2].std(), dfes.iloc[:,3].mean(), dfes.iloc[:,3].std(), dfes.iloc[:,4].sum(),
                dfes.iloc[:,5].mean(), dfes.iloc[:,5].std(), dfes.iloc[:,6].mean(), dfes.iloc[:,6].std(), dfes.iloc[:,7].mean(), dfes.iloc[:,7].std(), dfes.iloc[:,8].sum(),
                dfes.iloc[:,9].mean(), dfes.iloc[:,9].std(), dfes.iloc[:,10].mean(), dfes.iloc[:,10].std(), dfes.iloc[:,11].mean(), dfes.iloc[:,11].std(), dfes.iloc[:,12].sum()
                ])

    dfs = pd.DataFrame(summary, columns=header_summary)
    dfs.to_csv(os.path.join(dir_pred, "results_summary_{}.csv".format(time.strftime("%Y%m%d_%H%M%S"))), index=False)

    if 'testset' not in dir_pred.split('/')[-2]:
        # Validation
        lst_name_gt = [os.path.basename(gt).split(".")[0] for gt in lst_gt if int(os.path.basename(gt).split("_")[0][-1]) == 0]

        new_res_zs_val = []
        for j, rz in enumerate(res_zs_val):
            new_res_zs_val += [ ['{}_z{}'.format(lst_name_gt[j], idx),] + r for idx, r in enumerate(rz)]
        dfd = pd.DataFrame(new_res_zs_val, columns=HEADER)
        dfd.to_csv(os.path.join(dir_pred, "results_validation_detailed_{}.csv".format(time.strftime("%Y%m%d_%H%M%S"))), index=False)

        res_val = [[n,] + r for r, n in zip(res_val, lst_name_gt)]
        df = pd.DataFrame(res_val, columns=HEADER)
        df.to_csv(os.path.join(dir_pred, "results_validation_{}.csv".format(time.strftime("%Y%m%d_%H%M%S"))), index=False)

        header_summary = ["Group", "Dice LV", "Std LV", "Mod Dice LV", "X Std LV", "Volume LV", "Std Vol LV", "Cum Err LV(ml)",
                        "Dice RV", "Std RV", "Mod Dice RV", "X Std RV", "Volume RV", "Std Vol RV", "Cum Err RV(ml)",
                        "Dice MYO", "Std MYO", "Mod Dice MYO", "X Std MYO", "Volume MYO", "Std Vol MYO", "Cum Err MYO(ml)"]
        summary  = [['Total', 
                    df.iloc[:,1].mean(), df.iloc[:,1].std(), df.iloc[:,2].mean(), df.iloc[:,2].std(), df.iloc[:,3].mean(), df.iloc[:,3].std(), df.iloc[:,4].sum(),
                    df.iloc[:,5].mean(), df.iloc[:,5].std(), df.iloc[:,6].mean(), df.iloc[:,6].std(), df.iloc[:,7].mean(), df.iloc[:,7].std(), df.iloc[:,8].sum(),
                    df.iloc[:,9].mean(), df.iloc[:,9].std(), df.iloc[:,10].mean(), df.iloc[:,10].std(), df.iloc[:,11].mean(), df.iloc[:,11].std(), df.iloc[:,12].sum()
                    ]]
        dfed = df[df['Name'].str.contains('_ED')]
        summary.append(['ED', 
                    dfed.iloc[:,1].mean(), dfed.iloc[:,1].std(), dfed.iloc[:,2].mean(), dfed.iloc[:,2].std(), dfed.iloc[:,3].mean(), dfed.iloc[:,3].std(), dfed.iloc[:,4].sum(),
                    dfed.iloc[:,5].mean(), dfed.iloc[:,5].std(), dfed.iloc[:,6].mean(), dfed.iloc[:,6].std(), dfed.iloc[:,7].mean(), dfed.iloc[:,7].std(), dfed.iloc[:,8].sum(),
                    dfed.iloc[:,9].mean(), dfed.iloc[:,9].std(), dfed.iloc[:,10].mean(), dfed.iloc[:,10].std(), dfed.iloc[:,11].mean(), dfed.iloc[:,11].std(), dfed.iloc[:,12].sum()
                    ])
        dfes = df[df['Name'].str.contains('_ES')]
        summary.append(['ES', 
                    dfes.iloc[:,1].mean(), dfes.iloc[:,1].std(), dfes.iloc[:,2].mean(), dfes.iloc[:,2].std(), dfes.iloc[:,3].mean(), dfes.iloc[:,3].std(), dfes.iloc[:,4].sum(),
                    dfes.iloc[:,5].mean(), dfes.iloc[:,5].std(), dfes.iloc[:,6].mean(), dfes.iloc[:,6].std(), dfes.iloc[:,7].mean(), dfes.iloc[:,7].std(), dfes.iloc[:,8].sum(),
                    dfes.iloc[:,9].mean(), dfes.iloc[:,9].std(), dfes.iloc[:,10].mean(), dfes.iloc[:,10].std(), dfes.iloc[:,11].mean(), dfes.iloc[:,11].std(), dfes.iloc[:,12].sum()
                    ])

        dfs = pd.DataFrame(summary, columns=header_summary)
        dfs.to_csv(os.path.join(dir_pred, "results_validation_summary_{}.csv".format(time.strftime("%Y%m%d_%H%M%S"))), index=False)


def main(path_gt, path_pred):
    """
    Main function to select which method to apply on the input parameters.
    """
    if os.path.isfile(path_gt) and os.path.isfile(path_pred):
        compute_metrics_on_files(path_gt, path_pred)
    elif os.path.isdir(path_gt) and os.path.isdir(path_pred):
        compute_metrics_on_directories(path_gt, path_pred)
    else:
        raise ValueError("The paths given needs to be two directories or two files.")


def plot_results(path_pred, model_name):
    """
    Plot results per slice, cardiac phase and label.
    """
    res_detail = sorted(list(glob.iglob(os.path.join(path_pred, 'results_detailed_????????_??????.csv'))))[-1]
    detail = pd.read_csv(res_detail)
    phases = list(set([d[2] for d in detail.Name.str.split('_')]))
    detail['Phase'] = [d[2] for d in detail.Name.str.split('_')]
    detail['Slice'] = [int(d[-1][1:]) for d in detail.Name.str.split('_')]
    detail['Patient'] = [d[0] for d in detail.Name.str.split('_')]
    mid_slice = detail.groupby('Patient').max()['Slice']//2
    detail['RelSlice'] = 0
    for i in mid_slice.index:
        detail.loc[detail['Patient'] == i, 'RelSlice'] = detail[detail['Patient'] == i]['Slice'] - mid_slice[i]
    detail['Patient_Phase'] = detail['Patient'] + '_' + detail['Phase']
    detail['Average'] = detail['Mod Dice LV'] + detail['Mod Dice RV'] + detail['Mod Dice MYO']
    detail['Position'] = 'Other'
    detail['Group'] = 'Unknown'

    for ph in phases:
        for pat in detail.Patient.unique():
            names = sorted(detail.query('Patient == "{}" and Phase == "{}"'.format(pat, ph)).Name)
            if len(names) == 0:
                continue
            detail.loc[detail.Name == names[0], 'Position'] = 'Base'
            nmid = names[len(names)//2]
            detail.loc[detail.Name == nmid, 'Position'] = 'Middle'
            detail.loc[detail.Name == names[-1], 'Position'] = 'Apex'

    # Boxplot of all patients and slices
    sns.set(style="ticks")
    sns.set(rc={'figure.figsize': (16,15)})
    paltt = ['m','g','y']
    cases = ['LV', 'RV', 'MYO']
    f, axes = plt.subplots(nrows=len(cases), ncols=len(phases), squeeze=False)
    f.suptitle('Model {}'.format(model_name))
    for j, ph in enumerate(phases):
        axes[0,j].set_title(ph)
        for i, case in enumerate(cases):
            sns.boxplot(x="Group", y="Mod Dice {}".format(case), palette=paltt, hue='Position',
                        data=detail.query("Phase == '{}' and Position != 'Other'".format(ph)),
                        ax=axes[i,j])
            sns.stripplot(x="Group", y="Mod Dice {}".format(case), palette=paltt, hue='Position',
                        data=detail.query("Phase == '{}' and Position != 'Other'".format(ph)),
                        ax=axes[i,j], dodge=True, linewidth=0.5)

    f.savefig(os.path.join(path_pred, 'boxplots.pdf'))

    # Lineplot of accuracy per slice relative to mid slice
    f2, axes = plt.subplots(nrows=len(cases), ncols=len(phases), squeeze=False)
    f2.suptitle('Model {}'.format(model_name))
    for j, ph in enumerate(phases):
        axes[0,j].set_title(ph)
        for i, case in enumerate(cases):
            sns.lineplot(x="RelSlice", y="Mod Dice {}".format(case), hue="Patient",
                        data=detail.query("Phase == '{}'".format(ph)), ax=axes[i,j])
            axes[i,j].set(xlabel='Distance to middle slice (negative is appex)', ylabel="Dice {}".format(case))

    f2.savefig(os.path.join(path_pred, 'lineplots.pdf'))

    detail.to_csv(os.path.join(path_pred, 'results_detailed_transformed.csv'))

    # Clean seaborn plot styles
    sns.set_style("white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to compute ACDC challenge metrics.")
    parser.add_argument("GT_IMG", type=str, help="Ground Truth image")
    parser.add_argument("PRED_IMG", type=str, help="Predicted image")
    args = parser.parse_args()
    main(args.GT_IMG, args.PRED_IMG)
