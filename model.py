from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization, Activation
from keras.layers import Add, Concatenate
from keras.datasets import mnist
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

input = Input(shape=(28, 28, 1))
x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(input)
x = BatchNormalization()(x)

x1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
x1 = BatchNormalization()(x1)

x1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x1)
x1 = BatchNormalization()(x1)

x = Add()([x, x1])
x = MaxPooling2D(pool_size=(2, 2))(x)

x1 = Conv2D(48, kernel_size=(3, 3), activation='relu')(x)
x1 = BatchNormalization()(x1)

x1 = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x1)
x1 = BatchNormalization()(x1)

x2 = MaxPooling2D(pool_size=(2, 2))(x)

x2 = Conv2D(48, kernel_size=(3, 3), activation='relu')(x2)
x2 = BatchNormalization()(x2)

x = Concatenate()([x1, x2])
x2 = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=input, outputs=output)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#plt.imshow(x_test[0])
x_train = np.expand_dims(x_train, axis=-1).astype(np.float)/255.0
x_test = np.expand_dims(x_test, axis=-1).astype(np.float)/255.0
#n.shape

model.save('model.h5')
model.summary()

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.05
)
validdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

datagen.fit(x_train)
validdatagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size=132),
                    steps_per_epoch=len(x_train)//32,
                    epochs=20,
                    validation_data=validdatagen.flow(x_test, y_test, batch_size=32),
                    validation_steps=len(x_test)//32)

model.save_weights('weights1.h5')
