"""
Description: CNN training using adc images from ProstateX challenge.
"""

import keras
from keras import layers
from keras import models
from keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import plot_model


def main():
  
  # LOADING THE DATA
  adc_samples = np.load('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/numpy/ktrans/X_train.npy')
  adc_labels = np.load('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/numpy/ktrans/Y_train.npy')


  # CONVERT IMAGE SAMPLES TO FLOAT32 (REDUCE PRECISION FROM FLOAT64)
  adc_samples_flt32 = np.array(adc_samples, dtype=np.float32, copy = True)

  # RESHAPE IMAGE SAMPLES TO INCLUDE A SINGLE CHANNEL
  X = adc_samples_flt32.reshape((459,8,8,1))
  y = adc_labels
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

  epochs=100
  batch_size=80


  model = models.Sequential()
  model.add(layers.Conv2D(32, kernel_size=(3,3), padding = 'same', activation='relu', input_shape=(8,8,1)))
  model.add(layers.Conv2D(32, (3,3), activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Dropout(0.5))
  model.add(layers.Flatten())
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(512, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(256, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(128, kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001), activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1, activation='sigmoid'))
  model.summary()

  # COMPILATION
  opt = keras.optimizers.Adadelta()
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  # FIT

  history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.20, shuffle=True)

  model.save("/content/drive/My Drive/ProstrateX2/Saved Models/ktrans_cnn.h5")
  plot_model(model, to_file='/content/drive/My Drive/ProstrateX2/Diagrams/ktrans_cnn.png',show_shapes = True)
  
  print("model saved to disk")

  import matplotlib.pyplot as plt
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()


  eval=model.evaluate(X_test,y_test)
  print(eval)
  pred=model.predict(X_test)

  from sklearn.metrics import roc_curve, auc,roc_auc_score
  fpr, tpr, thresholds = roc_curve(y_test, pred)

  roc_auc = auc(fpr, tpr)

  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.show()
  pred=np.where(pred > 0.5, 1, 0)
  pred=pred[:,0]
  import pandas as pd
  results=pd.DataFrame({"Actual":y_test,"Predicted":pred})
  print(results)


  
  print("confusion_matrix : \n",confusion_matrix(pred,y_test))
  print("classification_report : \n",classification_report(pred,y_test))
  print("accuracy_score : \n",accuracy_score(pred,y_test))
  