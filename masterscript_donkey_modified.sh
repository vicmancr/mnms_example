#!/bin/sh

singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/victor_campello/mnms/network.lic -nodisplay -nosplash -batch "loadMMdata('/home/victor_campello/mnms/validation','/home/victor_campello/mnms/results/donkey/tmp/DATA.mat');exit"
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/victor_campello/mnms/network.lic -nodisplay -nosplash -batch "preprocessing_NN1st('/home/victor_campello/mnms/results/donkey/tmp/DATA.mat','/home/victor_campello/mnms/results/donkey/tmp/NN1st_input.mat');exit" 
singularity run --nv images/donkeyDANN2/my_python_container_DANN2.sif 1 /home/victor_campello/mnms/validation /home/victor_campello/mnms/results/donkey
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/victor_campello/mnms/network.lic -nodisplay -nosplash -batch "preprocessing_NN2nd('/home/victor_campello/mnms/results/donkey/tmp/DATA.mat','/home/victor_campello/mnms/results/donkey/tmp/prdNN1st.mat','/home/victor_campello/mnms/results/donkey/tmp/transformations.mat','/home/victor_campello/mnms/results/donkey/tmp/NN2nd_input.mat');exit" 
singularity run --nv images/donkeyDANN2/my_python_container_DANN2.sif 2 /home/victor_campello/mnms/validation /home/victor_campello/mnms/results/donkey
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/victor_campello/mnms/network.lic -nodisplay -nosplash -batch "postprocessing_NN2nd('/home/victor_campello/mnms/results/donkey/tmp/DATA.mat','/home/victor_campello/mnms/results/donkey/tmp/prdNN2nd.mat','/home/victor_campello/mnms/results/donkey/tmp/transformations.mat','/home/victor_campello/mnms/results/donkey');exit" 

singularity exec images/pollito.sif python metrics_mnms.py gt_validation results/donkey



singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "loadMMdata('/home/vec/Desktop/UPF/mnms_private/test1/mnms','/home/vec/Desktop/UPF/mnms_private/results/donkey/tmp/DATA.mat');exit"
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "preprocessing_NN1st('/home/vec/Desktop/UPF/mnms_private/results/donkey/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey/tmp/NN1st_input.mat');exit" 
singularity run --nv images/donkey/my_python_container_ensemble.sif 1 /home/vec/Desktop/UPF/mnms_private/validation /home/vec/Desktop/UPF/mnms_private/results/donkey
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "preprocessing_NN2nd('/home/vec/Desktop/UPF/mnms_private/results/donkey/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey/tmp/prdNN1st.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey/tmp/transformations.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey/tmp/NN2nd_input.mat');exit" 
singularity run --nv images/donkey/my_python_container_ensemble.sif 2 /home/vec/Desktop/UPF/mnms_private/validation /home/vec/Desktop/UPF/mnms_private/results/donkey
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "postprocessing_NN2nd('/home/vec/Desktop/UPF/mnms_private/results/donkey/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey/tmp/prdNN2nd.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey/tmp/transformations.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey');exit" 

singularity exec images/pollito.sif python metrics_mnms.py gt_validation results/donkey



sshfs -o workaround=rename bsc39304@plogin1.bsc.es:/home/bsc39/bsc39304/mnms/images mnms_images/


sudo rsync -chavzP --stats /home/vec/Desktop/UPF/mnms_private/images/ bsc39304@plogin1.bsc.es:/home/bsc39/bsc39304/mnms/images/



singularity run --nv images/donkey/my_python_container_ensemble2.sif 1 /home/cristian/Desktop/victor_challenge/validation /home/cristian/Desktop/victor_challenge/results/donkey
singularity run --nv images/donkey/my_python_container_ensemble2.sif 2 /home/cristian/Desktop/victor_challenge/validation /home/cristian/Desktop/victor_challenge/results/donkey



mkdir -p results/donkey1/tmp
mkdir -p results/donkey2/tmp
mkdir -p results/donkey3/tmp
mkdir -p results/donkey4/tmp
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "loadMMdata('/home/vec/Desktop/UPF/mnms_private/test1/mnms','/home/vec/Desktop/UPF/mnms_private/results/donkey1/tmp/DATA.mat');exit"
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "loadMMdata('/home/vec/Desktop/UPF/mnms_private/test2/mnms','/home/vec/Desktop/UPF/mnms_private/results/donkey2/tmp/DATA.mat');exit"
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "loadMMdata('/home/vec/Desktop/UPF/mnms_private/test3/mnms','/home/vec/Desktop/UPF/mnms_private/results/donkey3/tmp/DATA.mat');exit"
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "loadMMdata('/home/vec/Desktop/UPF/mnms_private/test4/mnms','/home/vec/Desktop/UPF/mnms_private/results/donkey4/tmp/DATA.mat');exit"
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "preprocessing_NN1st('/home/vec/Desktop/UPF/mnms_private/results/donkey1/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey1/tmp/NN1st_input.mat');exit" 
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "preprocessing_NN1st('/home/vec/Desktop/UPF/mnms_private/results/donkey2/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey2/tmp/NN1st_input.mat');exit" 
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "preprocessing_NN1st('/home/vec/Desktop/UPF/mnms_private/results/donkey3/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey3/tmp/NN1st_input.mat');exit" 
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "preprocessing_NN1st('/home/vec/Desktop/UPF/mnms_private/results/donkey4/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey4/tmp/NN1st_input.mat');exit" 
singularity run --nv images/donkey/my_python_container_ensemble.sif 1 /home/vec/Desktop/UPF/mnms_private/test1 /home/vec/Desktop/UPF/mnms_private/results/donkey1
singularity run --nv images/donkey/my_python_container_ensemble.sif 1 /home/vec/Desktop/UPF/mnms_private/test2 /home/vec/Desktop/UPF/mnms_private/results/donkey2
singularity run --nv images/donkey/my_python_container_ensemble.sif 1 /home/vec/Desktop/UPF/mnms_private/test3 /home/vec/Desktop/UPF/mnms_private/results/donkey3
singularity run --nv images/donkey/my_python_container_ensemble.sif 1 /home/vec/Desktop/UPF/mnms_private/test4 /home/vec/Desktop/UPF/mnms_private/results/donkey4
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "preprocessing_NN2nd('/home/vec/Desktop/UPF/mnms_private/results/donkey1/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey1/tmp/prdNN1st.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey1/tmp/transformations.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey1/tmp/NN2nd_input.mat');exit" 
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "preprocessing_NN2nd('/home/vec/Desktop/UPF/mnms_private/results/donkey2/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey2/tmp/prdNN1st.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey2/tmp/transformations.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey2/tmp/NN2nd_input.mat');exit" 
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "preprocessing_NN2nd('/home/vec/Desktop/UPF/mnms_private/results/donkey3/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey3/tmp/prdNN1st.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey3/tmp/transformations.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey3/tmp/NN2nd_input.mat');exit" 
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "preprocessing_NN2nd('/home/vec/Desktop/UPF/mnms_private/results/donkey4/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey4/tmp/prdNN1st.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey4/tmp/transformations.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey4/tmp/NN2nd_input.mat');exit" 
singularity run --nv images/donkey/my_python_container_ensemble.sif 2 /home/vec/Desktop/UPF/mnms_private/test1 /home/vec/Desktop/UPF/mnms_private/results/donkey1
singularity run --nv images/donkey/my_python_container_ensemble.sif 2 /home/vec/Desktop/UPF/mnms_private/test2 /home/vec/Desktop/UPF/mnms_private/results/donkey2
singularity run --nv images/donkey/my_python_container_ensemble.sif 2 /home/vec/Desktop/UPF/mnms_private/test3 /home/vec/Desktop/UPF/mnms_private/results/donkey3
singularity run --nv images/donkey/my_python_container_ensemble.sif 2 /home/vec/Desktop/UPF/mnms_private/test4 /home/vec/Desktop/UPF/mnms_private/results/donkey4
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "postprocessing_NN2nd('/home/vec/Desktop/UPF/mnms_private/results/donkey1/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey1/tmp/prdNN2nd.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey1/tmp/transformations.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey1');exit" 
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "postprocessing_NN2nd('/home/vec/Desktop/UPF/mnms_private/results/donkey2/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey2/tmp/prdNN2nd.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey2/tmp/transformations.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey2');exit" 
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "postprocessing_NN2nd('/home/vec/Desktop/UPF/mnms_private/results/donkey3/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey3/tmp/prdNN2nd.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey3/tmp/transformations.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey3');exit" 
singularity exec --nv --pwd /matlab_scripts images/donkey/my_matlab_container.sif matlab -c /home/vec/Desktop/UPF/mnms_private/network.lic -nodisplay -nosplash -batch "postprocessing_NN2nd('/home/vec/Desktop/UPF/mnms_private/results/donkey4/tmp/DATA.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey4/tmp/prdNN2nd.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey4/tmp/transformations.mat','/home/vec/Desktop/UPF/mnms_private/results/donkey4');exit" 

mkdir -p results/donkey
cp -r results/donkey1/*.nii.gz results/donkey
cp -r results/donkey2/*.nii.gz results/donkey
cp -r results/donkey3/*.nii.gz results/donkey
cp -r results/donkey4/*.nii.gz results/donkey

singularity exec --nv images/pollito.sif python metrics_mnms.py gt_test results/donkey




singularity run --nv images/donkey/my_python_container_ensemble.sif 2 /home/cristian/Desktop/victor_challenge/test1 /home/cristian/Desktop/victor_challenge/results/donkey1
singularity run --nv images/donkey/my_python_container_ensemble.sif 2 /home/cristian/Desktop/victor_challenge/test2 /home/cristian/Desktop/victor_challenge/results/donkey2
singularity run --nv images/donkey/my_python_container_ensemble.sif 2 /home/cristian/Desktop/victor_challenge/test3 /home/cristian/Desktop/victor_challenge/results/donkey3
singularity run --nv images/donkey/my_python_container_ensemble.sif 2 /home/cristian/Desktop/victor_challenge/test4 /home/cristian/Desktop/victor_challenge/results/donkey4



singularity exec --nv images/pollito.sif python metrics_mnms.py gt_validation val_results/seagull-V2/preds
singularity exec --nv images/pollito.sif python metrics_mnms.py gt_validation val_results/seagull-V3/preds
singularity exec --nv images/pollito.sif python metrics_mnms.py gt_validation val_results/seagull-V4/preds
singularity exec --nv images/pollito.sif python metrics_mnms.py gt_validation val_results/seagull-V5/preds

