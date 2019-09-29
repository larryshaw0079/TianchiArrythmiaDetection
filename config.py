# Dataset scale
INPUT_CHANNELS = 12
HIDDEN_CHANNELS = 64
NUM_CLASSES = 55

# Model training
LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY = 10
LEARNING_RATE_ADJUST = [32, 64, 128]
TRAIN_SPLIT = 0.9
BATCH_SIZE = 20
EPOCHS = 200
NUM_WORKERS = 14

# Model configuration
MODEL = 'inception'
assert(MODEL in ['resnet', 'inception'])
DILATED = True

# Helper
MULTI_GPU = False
SAVE_MODEL = True
SAVE_NAME = "inception_pre"
ENABLE_TENSORBOARD = True

MODE = 'noraml'
assert MODE in ['debug', 'restore', 'normal']
if MODE == 'debug':
    EPOCHS = 0
    SAVE_MODEL = False
    ENABLE_TENSORBOARD = False
elif MODE == 'restore':
    EPOCHS = 0
    SAVE_MODEL = False
    ENABLE_TENSORBOARD = False
    RESTORE_PATH = 'output/resnet_dilated_epoch30_2019-09-25 15:39:09.584330.pkl'
else:
    pass
