# Create an early stopping callback

import tensorflow as tf
def early_stopping(monitor='val_loss', min_delta=0.05, patience=3, verbose=1, mode='auto', restore_best_weights=True):
  """
  Creates an early_stopping element to be then used in the callback section when fitting a model.
  """
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                    min_delta=min_delta,
                                                    patience=patience,
                                                    verbose=verbose, 
                                                    mode=mode,
                                                    restore_best_weights=restore_best_weights)
  return early_stopping

# Unzip .zip file
import zipfile
def unzip(filepath):
  with zipfile.Zipfile('filepath','r') as zipObj:
    zipObj.extractall()
 
