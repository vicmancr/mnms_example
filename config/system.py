import os
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

### SET THESE PATHS MANUALLY #####################################################
# Full paths are required because otherwise the code will not know where to look
# when it is executed on one of the clusters.
at_cluster = False
project_root = '/mnsm_example'
if os.environ.get('LOGNAME') is not None:
    at_cluster = True  # Are you running this code on the cluster
    project_root = '/home/mnsm_example'

data_base = os.path.join(project_root, 'data')
log_root = os.path.join(project_root, 'models')
