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
from csutils.segmentation_summary import summary



parser = argparse.ArgumentParser(
    description="Script to evaluate a neural network model on the ACDC challenge data"
)
parser.add_argument('-m', "--model_name", type=str, default='unet2D_bn_modified_ds_mnms_rot', help="Name of experiment to use")
parser.add_argument('-t', '--evaluate_test_set', action='store_true')
parser.add_argument('-a', '--evaluate_all', action='store_true')
parser.add_argument('-i', '--iter', type=int, help='which iteration to use')
parser.add_argument('-d', '--dataset', type=str, default='mnms', help='Select which dataset to evaluate.')


if __name__ == "__main__":
    args = parser.parse_args()

    evaluate_test_set = args.evaluate_test_set
    evaluate_all = args.evaluate_all
    dataset = args.dataset
    all_datasets = [f for f in next(os.walk(sys_config.data_base))[1] if f[:2] != '__']
    if '-' in dataset:
        datasets = dataset.split('-')
        logging.info('Evaluating more than one dataset at the same time: {0}'.format(datasets))
        for ds in datasets:
            assert ds in all_datasets, 'Dataset "{0}" not available. \
                Please, choose one of the following: {1}'.format(ds, all_datasets)
    else:
        assert dataset in all_datasets, 'Dataset "{0}" not available. \
            Please, choose one of the following: {1}'.format(dataset, all_datasets)

    if evaluate_test_set and evaluate_all:
        raise ValueError('evaluate_all and evaluate_test_set cannot be chosen together!')

    use_iter = args.iter
    if use_iter:
        logging.info('Using iteration: {0}'.format(use_iter))


    base_path = sys_config.project_root
    model_path = os.path.join(sys_config.log_root, args.model_name)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')
    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    if evaluate_test_set:
        logging.warning('EVALUATING ON TEST SET')
        output_path = os.path.join(model_path, dataset, 'predictions_testset')
    elif evaluate_all:
        logging.warning('EVALUATING ON ALL DATA')
        output_path = []
        output_path.append(os.path.join(model_path, dataset, 'predictions'))
        output_path.append(os.path.join(model_path, dataset, 'predictions_testset'))
    else:
        logging.warning('EVALUATING ON VALIDATION SET')
        output_path = os.path.join(model_path, dataset, 'predictions')



    model = Model(exp_config)
    database = dataset if '-' not in dataset else datasets
    if evaluate_all:

        for outp, subset in zip(output_path, ['training', 'testing']):
            path_pred = os.path.join(outp, 'prediction')
            path_image = os.path.join(outp, 'image')
            utils_gen.makefolder(path_pred)
            utils_gen.makefolder(path_image)

            path_gt = os.path.join(outp, 'ground_truth')
            path_diff = os.path.join(outp, 'difference')
            path_eval = os.path.join(outp, 'eval')
            utils_gen.makefolder(path_diff)
            utils_gen.makefolder(path_gt)

            init_iteration = evaluation(model,
                                        outp,
                                        model_path,
                                        database,
                                        subset,
                                        exp_config=exp_config,
                                        do_postprocessing=True,
                                        evaluate_all=True,
                                        use_iter=use_iter)

            metrics_acdc.main(path_gt, path_pred)
            metrics_acdc.plot_results(path_pred)

        summary(args.model_name)

    else:

        path_pred = os.path.join(output_path, 'prediction')
        path_image = os.path.join(output_path, 'image')
        utils_gen.makefolder(path_pred)
        utils_gen.makefolder(path_image)

        path_gt = os.path.join(output_path, 'ground_truth')
        path_diff = os.path.join(output_path, 'difference')
        path_eval = os.path.join(output_path, 'eval')
        utils_gen.makefolder(path_diff)
        utils_gen.makefolder(path_gt)

        subset = 'testing' if evaluate_test_set else 'training'
        init_iteration = evaluation(model,
                                    output_path,
                                    model_path,
                                    database,
                                    subset,
                                    exp_config=exp_config,
                                    do_postprocessing=True,
                                    evaluate_all=True,
                                    use_iter=use_iter)

        metrics_acdc.main(path_gt, path_pred)
        metrics_acdc.plot_results(path_pred)

        summary(args.model_name)
