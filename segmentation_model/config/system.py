import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

home = os.environ.get('HOME')
project_root = os.path.join(home, 'mnms_example')
if os.environ.get('SINGULARITY_NAME') is not None:
    project_root = os.path.join('/gpfs/projects/bsc39/bsc39304/mnms')

data_base = os.path.join(project_root, 'data')
log_root = os.path.join(project_root, 'models')
