# config.py

import os

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2

# File paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
UTILS_DIR = 'utils'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'neural_network.h5')

# Other settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
