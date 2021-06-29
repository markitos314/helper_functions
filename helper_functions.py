# Create an early stopping callback

import tensorflow as tf
import matplotlib.pyplot as plt

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


def predict_and_plot(model, filepath, img_size=224, scale=False, class_names=[]):
  """
  Function that loads an image, processes it and prints it

  Args:
    model: TensorFlow type model necessary to predict.
    filepath (str): filepath of the file to load.
    img_size (int): size of the output image after tensorflows processing.
                    WARNING: should be the same size as the models trained 
                    image sizes.
    class_names (array of str): array with all the class names in the model.
    class_truth (str): truth label of the loaded (for prediction) image.
  """
  # Load image to tensorflow
  img = tf.io.read_file(filepath)
  # Convert to tensor
  img = tf.image.decode_image(img, channels=3)
  # Resize
  img = tf.image.resize(img, [img_size,img_size])
  # Rescale if necesary
  if scale:
    img = img/255.
  # Expand dimensions for prediction
  img = tf.expand_dims(img, axis=0)
  # Make predictions
  preds = model.predict(img)
  # Transform predictions to words
  class_name = class_names[preds.argmax()]
  # Squeeze image for plotting
  img = tf.squeeze(img)
  # Plot image
  plt.imshow(img/255.)
  plt.axis(False)
  plt.title(f'Pred: {class_name}, Confidence: {preds.max()*100:.2f}%')
  
# Build data augmentation sequence
def data_augmentation_layer()
  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
    tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)
  ])
  return data_augmentation

# Make model checkpoint callback
def make_model_checkpoint(filepath)
  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                        monitor='val_loss',
                                                        verbose=1,
                                                        save_best_only=True)
  return model_checkpoint


