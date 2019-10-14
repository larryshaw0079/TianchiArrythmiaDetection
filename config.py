# Dataset scale
INPUT_CHANNELS = 12
HIDDEN_CHANNELS = 64
NUM_CLASSES = 55

# Model training
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = 10
LEARNING_RATE_ADJUST = [150, 300]
TRAIN_SPLIT = 0.9
BATCH_SIZE = 20
EPOCHS = 250
EPOCH_TO_CHANGE = 250
NUM_WORKERS = 14
SEED = 2019
CRITERION = 'focal'
assert(CRITERION in ['bce', 'focal'])

# Model configuration
MODEL = 'resnet'
assert(MODEL in ['resnet', 'inception', 'densenet'])
USE_SE = False
USE_SPECTRAL = False

# Focal loss setting
FOCAL_GAMMA = 2

# Helper
MULTI_GPU = False
SAVE_MODEL = True
SAVE_NAME = "resnet_valinna_focal"
ENABLE_TENSORBOARD = False

MODE = 'restore'
assert MODE in ['debug', 'restore', 'normal']
if MODE == 'debug':
    EPOCHS = 1
    EPOCH_TO_CHANGE = 1
    SAVE_MODEL = False
    ENABLE_TENSORBOARD = False
elif MODE == 'restore':
    RESUME_EPOCH = 250
    SAVE_MODEL = False
    ENABLE_TENSORBOARD = False
    RESTORE_PATH = 'output/parameters/param_resnet_valinna_focal_epoch150.pkl'
else:
    pass
