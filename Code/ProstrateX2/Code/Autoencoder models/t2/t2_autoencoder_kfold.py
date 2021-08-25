
import numpy as np
from sklearn.model_selection import train_test_split

# LOADING THE DATA
train_t2_samples = np.load('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/numpy/t2/X_train.npy')
train_t2_labels = np.load('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/numpy/t2/Y_train.npy')

t2_samples_flt32 = np.array(train_t2_samples, dtype=np.float32, copy = True)
# RESHAPE IMAGE SAMPLES TO INCLUDE A SINGLE CHANNEL

X = t2_samples_flt32.reshape((459,32,32,1))
y = train_t2_labels


from keras.models import Model,load_model
from  keras.layers import Flatten,Dense,Dropout
from keras.optimizers import Adadelta
from keras.layers import Conv2D,MaxPooling2D
from keras import regularizers

encode=load_model("/content/drive/My Drive/ProstrateX2/Saved Models/t2_autoencoder.h5")
print("Loaded model from disk")
for i in range(15):
    encode.layers.pop()
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




#KFOLD
from sklearn.model_selection import KFold
cvscores = []
kfold = KFold(n_splits=5, shuffle=True, random_state=7)

for train, test in kfold.split(X, y):
    full_model = Model(encode.inputs,fc(encode.get_layer("encoder").output))
    for layer in full_model.layers[0:19]:
        layer.trainable = False
    opt = Adadelta()
    full_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    classify_train = full_model.fit(X[train],y[train], batch_size=64, epochs=100,  verbose=0,shuffle=True)
    scores = full_model.evaluate(X[test], y[test], verbose=0)
    print("%s: %.2f%%" % (full_model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
print("avg accurecy :", np.mean(cvscores))
