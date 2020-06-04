# -*- coding: utf-8 -*-
import os
import sys
import glob
import logging
import argparse
from importlib.machinery import SourceFileLoader

import config.system as sys_config
from utils import utils_gen

wd = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(wd, 'cardiac-segmentation'))
from model import Model
from evaluate_patients import score_data as evaluation
import metrics_acdc



parser = argparse.ArgumentParser(
    description="Example script to evaluate a model on the M&Ms challenge data"
)
parser.add_argument("in_datapath", type=str, help="Path to dataset")
parser.add_argument("out_datapath", type=str, help="Output path to generated files")
parser.add_argument('-m', "--model_name", type=str, default='unet2D_bn_modified_ds_mnms_rot', help="Name of experiment to use")


if __name__ == "__main__":
    args = parser.parse_args()

    dataset = 'mnms'
    in_datapath = os.path.abspath(args.in_datapath)
    out_datapath = os.path.abspath(args.out_datapath)

    base_path = sys_config.project_root
    model_path = os.path.join(sys_config.log_root, args.model_name)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    logging.warning('EVALUATING ON TEST SET')
    output_path = os.path.join(out_datapath, dataset, 'predictions')


    model = Model(exp_config)

    path_pred = os.path.join(output_path, 'prediction')
    path_image = os.path.join(output_path, 'image')
    utils_gen.makefolder(path_pred)
    utils_gen.makefolder(path_image)

    path_gt = os.path.join(output_path, 'ground_truth')
    path_diff = os.path.join(output_path, 'difference')
    path_eval = os.path.join(output_path, 'eval')
    utils_gen.makefolder(path_diff)
    utils_gen.makefolder(path_gt)

    init_iteration = evaluation(model,
                                output_path,
                                in_datapath,
                                model_path,
                                dataset,
                                exp_config=exp_config,
                                do_postprocessing=True)
