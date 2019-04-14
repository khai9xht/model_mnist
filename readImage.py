from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model.h5')
model.load_weights('weights1.h5')
#model.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#plt.imshow(x_test[0])
x_train = np.expand_dims(x_train, axis=-1).astype(np.float)/255.0
#x_test = np.expand_dims(x_test, axis=-1).astype(np.float)/255.0
#x_train.shape

validdatagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)
validdatagen.fit(x_train)

image = cv2.imread("image/nb7.jpg")
im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
im, thre = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
thre.shape
#plt.imshow(thre)
thre = np.expand_dims(thre, axis=-1).astype(np.float)/255.0
results = model.predict_generator(
    validdatagen.flow(np.array([thre]), batch_size=1, shuffle=False),
    steps=1
)
y_pred = np.argmax(results, axis=-1)
print(y_pred)
