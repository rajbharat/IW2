import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def main():
  # LOADING THE DATA
  train_t2_samples = np.load('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/numpy/adc/X_train.npy')
  train_t2_labels = np.load('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/numpy/adc/Y_train.npy')

  t2_samples_flt32 = np.array(train_t2_samples, dtype=np.float32, copy = True)
  # RESHAPE IMAGE SAMPLES TO INCLUDE A SINGLE CHANNEL

  x_train = t2_samples_flt32.reshape((459,8,8,1))
  y_train = train_t2_labels


  from keras.models import Model,load_model
  from  keras.layers import Flatten,Dense,Dropout
  from keras.optimizers import Adadelta
  from keras.layers import Conv2D,MaxPooling2D
  from keras import regularizers


  X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42)


  encode=load_model("/content/drive/My Drive/ProstrateX2/Saved Models/adc_autoencoder.h5")
  print("Loaded model from disk")

  for i in range(15):
      encode.layers.pop()
  val_X=np.concatenate((X_train[:64],X_test[:10]))
  val_y=np.concatenate((y_train[:64],y_test[:10]))

  print(encode.summary())

  #Classification
  def fc(enco):
      x=Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',name='a')(enco)
      x=Dropout(0.3)(x)
      x=MaxPooling2D((2, 2), padding='same',name='b')(x)
      x=Conv2D(64, (3, 3), activation='relu', padding='same',name='c')(x)
      x=Dropout(0.3)(x)
      x=MaxPooling2D(pool_size=(2, 2), padding='same',name='d')(x)
      x=Conv2D(128, (3, 3), activation='relu', padding='same',name='e')(x)
      x=Dropout(0.3)(x)
      x=MaxPooling2D(pool_size=(2, 2), padding='same',name='f')(x)
      x=Flatten()(x)
      x=Dense(128, activation='relu',name='g',kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001))(x)
      x=Dropout(0.3)(x)
      out = Dense(1, activation='sigmoid',name='h')(x)
      return out


  full_model = Model(encode.inputs,fc(encode.get_layer("encoder").output))

  for layer in full_model.layers[0:19]:
      layer.trainable = False

  full_model.summary()

  opt =Adadelta()
  full_model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
  classify_train = full_model.fit(X_train, y_train, batch_size=64,epochs=100,validation_data=(val_X,val_y),verbose=1,shuffle=True)
  full_model.save_weights('/content/drive/My Drive/ProstrateX2/Saved Models/adc_autoencoder_classification.h5')
  plot_model(full_model, to_file='/content/drive/My Drive/ProstrateX2/Diagrams/adc_autoencoder_classification.png',show_shapes = True)
  
  import matplotlib.pyplot as plt
  plt.plot(classify_train.history['acc'])
  plt.plot(classify_train.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(classify_train.history['loss'])
  plt.plot(classify_train.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()


  test_eval = full_model.evaluate(X_test, y_test, verbose=0)
  print('Test loss:', test_eval[0])
  print('Test accuracy:', test_eval[1])

  pred=full_model.predict(X_test)

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
  


