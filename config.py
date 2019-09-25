# Dataset scale
INPUT_CHANNELS = 12
HIDDEN_CHANNELS = 64
NUM_CLASSES = 55

# Model training
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.9
BATCH_SIZE = 20
EPOCHS = 30
NUM_WORKERS = 28

# Model configuration
DILATED = True

# Helper
MULTI_GPU = True
SAVE_MODEL = True
SAVE_NAME = "resnet_dilated"
ENABLE_TENSORBOARD = True
