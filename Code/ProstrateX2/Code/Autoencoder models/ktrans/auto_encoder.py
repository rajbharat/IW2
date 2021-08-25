import numpy as np

# LOADING THE DATA
train_t2_samples = np.load('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/numpy/ktrans/X_train.npy')
train_t2_labels = np.load('/content/drive/My Drive/ProstrateX2/Dataset/Training/generated/numpy/ktrans/Y_train.npy')
t2_samples_flt32 = np.array(train_t2_samples, dtype=np.float32, copy = True)
# RESHAPE IMAGE SAMPLES TO INCLUDE A SINGLE CHANNEL
x_train = t2_samples_flt32.reshape((459,8,8,1))



import numpy as np


y_train = train_t2_labels

from keras.layers import UpSampling2D
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input,Conv2D,BatchNormalization,MaxPooling2D

batch_size = 64
epochs = 200
input_img = Input(shape = (8,8,1))


def encoder(input_img):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization(name="encoder")(conv4)
    return conv4

def decoder(conv4):
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2, 2))(conv6)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2, 2))(conv7)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)
    return decoded

def main():
    autoencoder = Model(input_img, decoder(encoder(input_img)))
    autoencoder.summary()
    autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
    autoencoder_train = autoencoder.fit(x_train, x_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2)
    model_json = autoencoder.to_json()
    autoencoder.save("/content/drive/My Drive/ProstrateX2/Saved Models/ktrans_autoencoder.h5")
    print("Saved model")


    import matplotlib.pyplot as plt
    plt.plot(autoencoder_train.history['loss'])
    plt.plot(autoencoder_train.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")

