import os
import glob, cv2
import numpy as np
from UNet import UNet
import segmentation_models as sm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.optimizers import Adam
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import jaccard_score

seed = 24
batch_size = 16
from keras.preprocessing.image import ImageDataGenerator

img_data_gen_args = dict(rescale = 1/255.,
                         rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')

mask_data_gen_args = dict(rescale = 1/255.,  #Original pixel values are 0 and 255. So rescaling to 0 to 1
                        rotation_range=90,
                      width_shift_range=0.3,
                      height_shift_range=0.3,
                      shear_range=0.5,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='reflect')

val_img_data_gen_args = dict(rescale = 1/255.)

image_data_generator = ImageDataGenerator(**img_data_gen_args)
image_generator = image_data_generator.flow_from_directory('/kaggle/input/dsb256x256/train-images/train-images', 
                                                           seed=seed,
                                                           shuffle=True,
                                                           target_size=(256,256),
                                                           batch_size=batch_size,
                                                           class_mode=None)  #Very important to set this otherwise it returns multiple numpy arrays 
                                                                             #thinking class mode is binary.

mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
mask_generator = mask_data_generator.flow_from_directory('/kaggle/input/dsb256x256/train-masks/train-masks', 
                                                         seed=seed, 
                                                         shuffle=True,
                                                         target_size=(256,256),
                                                         batch_size=batch_size,
                                                         color_mode = 'grayscale',   #Read masks in grayscale
                                                         class_mode=None)

val_image_data_generator = ImageDataGenerator(**val_img_data_gen_args)
valid_img_generator = val_image_data_generator.flow_from_directory('/kaggle/input/dsb256x256/val-images/val-images', 
                                                               seed=seed,
                                                               target_size=(256,256),
                                                               batch_size=batch_size, 
                                                               class_mode=None) #Default batch size 32, if not specified here

val_mask_data_generator = ImageDataGenerator(**val_img_data_gen_args)
valid_mask_generator = val_mask_data_generator.flow_from_directory('/kaggle/input/dsb256x256/val-masks/val-masks', 
                                                               seed=seed,
                                                               target_size=(256,256),
                                                               batch_size=batch_size, 
                                                               color_mode = 'grayscale',   #Read masks in grayscale
                                                               class_mode=None)  #Default batch size 32, if not specified here


train_generator = zip(image_generator, mask_generator)
val_generator = zip(valid_img_generator, valid_mask_generator)

x = image_generator.next()
y = mask_generator.next()

for i in range(4):
    ax = plt.subplot(2, 4, i + 1)
    plt.imshow((x[i]*255).astype("uint8"))
c = 4
for i in range(4):
    ax = plt.subplot(2, 4, c + i + 1)
    plt.imshow((y[i]*255).astype("uint8"), cmap='gray')
plt.show()

#Jaccard distance loss mimics IoU. 
from keras import backend as K
def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + focal_loss

IMG_HEIGHT = x.shape[1]
IMG_WIDTH  = x.shape[2]
IMG_CHANNELS = x.shape[3]
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = UNet()
model.compile(optimizer=Adam(0.00001, beta_1=0.99, beta_2=0.99), loss=[total_loss], metrics=[jaccard_coef])

num_train_imgs = 5520
callbacks = tf.keras.callbacks.ModelCheckpoint('/kaggle/working/mSMAD-{epoch}-{jaccard_coef}-{val_loss}.h5', period=5)

steps_per_epoch = num_train_imgs // batch_size

history = model.fit(train_generator, validation_data=val_generator,
                    batch_size=batch_size, validation_batch_size=batch_size, 
                    steps_per_epoch=steps_per_epoch, 
                    validation_steps=steps_per_epoch, epochs=250,
                    callbacks= [callbacks] )

model.save('/kaggle/working/model1.h5')

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()