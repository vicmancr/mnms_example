export nnUNet_raw_data_base=/gpfs/projects/bsc39/bsc39304/mnms/nnUNet_raw_data_base
export nnUNet_preprocessed=/gpfs/projects/bsc39/bsc39304/mnms/nnUNet_cropped_data
export RESULTS_FOLDER=/gpfs/projects/bsc39/bsc39304/mnms/results

nnUNet_train 2d nnUNetTrainerV2 1 0 -val
# nnUNet_train 2d nnUNetTrainerV2 1 1
# nnUNet_train 2d nnUNetTrainerV2 1 2
# nnUNet_train 2d nnUNetTrainerV2 1 3
# nnUNet_train 2d nnUNetTrainerV2 1 4
