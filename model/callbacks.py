from tensorflow import keras
import os
from utils.utils import get_run_logdir
import matplotlib.pyplot as plt
from utils.utils import DataGenerator
import numpy as np
import itertools

def callbacks(dir_name = None, model_name='model.h5', patience = 10):
    checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(dir_name, model_name), save_best_only = True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience = patience, restore_best_weights = True)
    tensorboard_cb = keras.callbacks.TensorBoard(os.path.join(dir_name, "tensorboard"), histogram_freq=1)

    return [checkpoint_cb, early_stopping_cb, tensorboard_cb]

 