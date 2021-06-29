# Create an early stopping callback

import tensorflow as tf
def early_stopping(monitor='val_loss', min_delta=0.05, patience=3, verbose=1, mode=auto, restore_best_weights=True):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor,
                                                    min_delta,
                                                    patience,
                                                    verbose, 
                                                    mode,
                                                    restore_best_weights)
  return early_stopping
