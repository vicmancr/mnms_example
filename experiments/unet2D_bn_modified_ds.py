import importlib
import tensorflow as tf

experiment_name = 'unet2D_bn_modified_ds'

# Model settings
model_handle = importlib.import_module('model_zoo.unet2D_bn_modified_ds').forward

# Data settings
data_mode = '2D'
model_type = 'convolutional'
data_file_mask = 'data_{0}_{1}{2}_size_{3}_res_{4}{5}.hdf5' # 0: dataset; 1: mode; 2: _subset; 3: size of image; 4: resolution; 5: _onlytrain
image_size = (256, 256)
label_size = (256, 256)
target_resolution = (1, 1)
nlabels = 4

# Training settings
batch_size = 1
learning_rate = 0.0001
optimizer_handle = tf.train.AdamOptimizer
schedule_lr = False
warmup_training = True
weight_decay = 0.00000
momentum = None
loss_type = 'weighted_crossentropy'  # crossentropy/weighted_crossentropy/dice/dice_onlyfg

# Augmentation settings
augment_batch = False
do_rotations = False
do_strong_rotations = False
do_scaleaug = False
do_filplr = False
do_shear = False
do_disturb = 0

# Rarely changed settings
train_on_all_data = False  # Use also test for training
use_data_fraction = False  # Should normally be False
max_epochs = 20000
schedule_gradient_threshold = 0.00001  # When the gradient of the learning curve is smaller than this value the LR will
                                       # be reduced

train_eval_frequency = 200
val_eval_frequency = 100
