import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

home = os.environ.get('HOME')
project_root = os.path.join(home, 'mnms_example')
if os.environ.get('USER') == 'bsc39304':
    data_root = os.path.join('/gpfs/projects/bsc39/bsc39304/mnms')
    project_root = os.path.join('/home/bsc39/bsc39304/mnms/segmentation_model')
else:
    project_root = os.path.join('/home/vec/Desktop/UPF/mnms_private/segmentation_model')
    data_root = project_root
    
data_base = os.path.join(data_root, 'data')
log_root = os.path.join(data_root, 'models')
