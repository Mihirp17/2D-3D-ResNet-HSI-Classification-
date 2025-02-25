import torch

class Config:
    # Dataset parameters
    DATASET = 'LO'  #
    NUM_COMPONENTS = 32
    SPATIAL_SIZE = 15
    TR_PERCENT = 0.15  # Increased training data percentage

    # Training parameters
    EPOCHS = 500 # Full training run
    LEARNING_RATE = 0.4
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    TRAIN_BATCH_SIZE = 128  # Increased for faster training
    TEST_BATCH_SIZE = 256

    # Model parameters
    DEPTH = 32
    ALPHA = 48
    INPLANES = 16
    BOTTLENECK = True

    # Validation
    USE_VAL = True
    VAL_PERCENT = 0.1

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths
    DATA_PATH = 'attached_assets'
    MODEL_SAVE_PATH = 'best_model.pth.tar'

    # Debug mode
    DEBUG = False