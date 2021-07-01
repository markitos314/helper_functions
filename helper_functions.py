import tensorflow as tf
import matplotlib.pyplot as plt
import zipfile

# Create an early stopping callback
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
def unzip(filepath):
  with zipfile.Zipfile('filepath','r') as zipObj:
    zipObj.extractall()

# Predict and plot image
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

# Preprocess DF for HMN
def preprocess_emergencias(path):
  """
  Function that preprocesses 'emergencias' csv from Pentaho.
  
  Args:
    path (str): Path to csv to preprocess

  Returns:
    Preprocessed pandas dataframe
  """
  # Read csv
  df_temp = pd.read_csv(path)

  # Get rid of unnecessary columns
  df_temp.drop(columns=[f'Unnamed: {i}' for i in range(18,27,1)], inplace=True)
  df_temp.drop(columns=['Unnamed: 0','Unnamed: 7','Unnamed: 10'], inplace=True)

  # Get rid of unnecessary rows
  df_temp.drop(range(0,6), inplace=True)

  # Set definitive columns
  columns = ['DNI', 'NHC', 'PACIENTE', 'SEXO', 'EDAD', 'FECHA_HORA_INGRESO', 
          'SERVICIO', 'SECCION', 'ALTA_MEDICA', 'MOTIVO_ALTA', 'ALTA_ADMIN', 
          'PROFESIONAL', 'DIAGNOSTICO', 'CIE10', 'DESC_CIE10']
  df_temp.columns=columns

  # Convert dates to datetype format
  df_temp['ALTA_ADMIN'] = pd.to_datetime(df_temp['ALTA_ADMIN'], dayfirst=True)
  df_temp['ALTA_MEDICA'] = pd.to_datetime(df_temp['ALTA_MEDICA'], dayfirst=True)
  df_temp['FECHA_HORA_INGRESO'] = pd.to_datetime(df_temp['FECHA_HORA_INGRESO'], dayfirst=True)

  # Create time difference columns
  df_temp['DIF_ALTA_ADMIN_MEDICA'] = df_temp['ALTA_ADMIN'] - df_temp['ALTA_MEDICA']
  df_temp['DIF_ALTA_MEDICA_INGRESO'] = df_temp['ALTA_MEDICA'] - df_temp['FECHA_HORA_INGRESO']
  df_temp['ESTADIA_TOTAL'] = df_temp['ALTA_ADMIN'] - df_temp['FECHA_HORA_INGRESO']

  # Sort by 'FECHA_HORA_INGRESO'
  df_temp.sort_values('FECHA_HORA_INGRESO', inplace=True)

  # Reset index
  df_temp.reset_index(drop=True, inplace=True)
  return df_temp


# Contatenate dfs
def concatenate_dfs(path, keyword):
  """
  Function that merges all preprocessed dfs in a directory, filtered by keyword

  Args:
    path (str): directory containing files to be preprocessed.
    keyword (str): string used to filter type of files.
  
  Returns:
    Concatenation of all pandas dataframes result of preprocessing all files in
    directory.
  """
  df_list = []
  for i, name in enumerate(glob.glob(path + '/*' + keyword + '*')):
    df_list.append(preprocess_emergencias(name))
    df = pd.concat(df_list)
  return df
