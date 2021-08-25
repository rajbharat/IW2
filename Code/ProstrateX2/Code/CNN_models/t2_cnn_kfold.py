import keras
from keras import layers
from keras import models
from keras import regularizers




import numpy as np


# LOADING THE DATA
train_t2_samples = np.load('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/numpy/t2/X_train.npy')
train_t2_labels = np.load('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/numpy/t2/Y_train.npy')
t2_samples_flt32 = np.array(train_t2_samples, dtype=np.float32, copy = True)
# RESHAPE IMAGE SAMPLES TO INCLUDE A SINGLE CHANNEL
X = t2_samples_flt32.reshape((459,32,32,1))
y = train_t2_labels


from sklearn.model_selection import KFold
cvscores = []
kfold = KFold(n_splits=5, shuffle=True, random_state=7)

for train, test in kfold.split(X, y):
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001), activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation='sigmoid'))
    # model.summary()

    opt = keras.optimizers.Adadelta()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    # FIT
    # validation_split = 0.25,, class_weight={0:1,1:2}
    history = model.fit(X[train], y[train], epochs=100, batch_size=64, shuffle=True,verbose=0)

    scores = model.evaluate(X[test],y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
print("avg accurecy :",np.mean(cvscores))

