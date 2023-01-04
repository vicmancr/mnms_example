export nnUNet_raw_data_base=/gpfs/projects/bsc39/bsc39304/mnms/nnUNet_raw_data_base
export nnUNet_preprocessed=/gpfs/projects/bsc39/bsc39304/mnms/nnUNet_cropped_data
export RESULTS_FOLDER=/gpfs/projects/bsc39/bsc39304/mnms/results

nnUNet_predict -chk model_best -f 0 -i /gpfs/projects/bsc39/bsc39304/mnms/nnUNet_raw_data_base/nnUNet_raw_data/val -o /gpfs/projects/bsc39/bsc39304/results/nnUNet/2d/Task001_Mnms/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/val_raw -t 1 -m 2d

nnUNet_predict -chk model_best -f 1 -i /gpfs/projects/bsc39/bsc39304/mnms/nnUNet_raw_data_base/nnUNet_raw_data/val -o /gpfs/projects/bsc39/bsc39304/results/nnUNet/2d/Task001_Mnms/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/val_raw -t 1 -m 2d

nnUNet_predict -chk model_best -f 2 -i /gpfs/projects/bsc39/bsc39304/mnms/nnUNet_raw_data_base/nnUNet_raw_data/val -o /gpfs/projects/bsc39/bsc39304/results/nnUNet/2d/Task001_Mnms/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/val_raw -t 1 -m 2d

nnUNet_predict -chk model_best -f 3 -i /gpfs/projects/bsc39/bsc39304/mnms/nnUNet_raw_data_base/nnUNet_raw_data/val -o /gpfs/projects/bsc39/bsc39304/results/nnUNet/2d/Task001_Mnms/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/val_raw -t 1 -m 2d

nnUNet_predict -chk model_best -f 4 -i /gpfs/projects/bsc39/bsc39304/mnms/nnUNet_raw_data_base/nnUNet_raw_data/val -o /gpfs/projects/bsc39/bsc39304/results/nnUNet/2d/Task001_Mnms/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/val_raw -t 1 -m 2d


