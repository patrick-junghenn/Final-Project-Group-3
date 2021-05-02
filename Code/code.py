from imutils import paths
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, AveragePooling2D, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2
import os


imagePaths = list(paths.list_images('/Users/patrickjunghenn/Desktop/covid_19_dataset/X-ray'))


data = []
labels = []

for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	data.append(image)
	labels.append(label)

data = np.array(data) / 255.0
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = tensorflow.keras.utils.to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.25, stratify=labels, random_state=42)

# image augmentation for training
train_aug = ImageDataGenerator(rotation_range=15, fill_mode='nearest')

input_shape = (224, 224, 3)
# pretrained on imagenet
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))

head_model = base_model.output
head_model = AveragePooling2D(pool_size=(4,4))(head_model)
head_model = Flatten()(head_model)
head_model = Dense(64, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation='softmax')(head_model)

# head_model placed on top of vgg16
model = Model(inputs=base_model.input, outputs=head_model)
for layer in base_model.layers:
    layer.trainable = False

# callbacks
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 5,
                              verbose = 1,
                              min_delta = 0.001)

earlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 5,
                          verbose = 1,
                          restore_best_weights = True)

callbacks = [reduce_lr, earlystop]

# compile model
n_epochs = 30
opt = Adam(lr=1e-3, decay= 1e-3/n_epochs)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# train full model
batch_size = 8
model_fitted = model.fit_generator(train_aug.flow(X_train,
                                                  y_train,
                                                  batch_size= batch_size),
                                   steps_per_epoch=len(X_train) // batch_size,
                                   validation_data=(X_test, y_test),
                                   validation_steps = len(X_test) // batch_size,
                                   callbacks = callbacks,
                                   epochs=n_epochs)


# predictions
pred = model.predict(X_test, batch_size=batch_size)
pred = np.argmax(pred, axis=1)

# model evaluation
print(classification_report(y_test.argmax(axis=1), pred))


# Plotting loss of model with pretrained weights
fitted_dict = model_fitted.history
loss_values = fitted_dict['loss']
val_loss_values = fitted_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
line2 = plt.plot(epochs, loss_values, label='Training Loss')
plt.setp(line1, linewidth=2.0, marker = '+', markersize=12.0)
plt.setp(line2, linewidth=2.0, marker = '4', markersize=12.0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Per Epoch: Pretrained Weights')
plt.grid(True)
plt.legend()
plt.show()
