import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

home = os.environ.get('HOME')
project_root = os.path.join(home, 'mnms_example/segmentation_model')
if os.environ.get('SINGULARITY_NAME') is not None:
    project_root = os.path.join('/mnms', 'segmentation_model')

data_base = os.path.join(project_root, 'data')
log_root = os.path.join(project_root, 'models')
