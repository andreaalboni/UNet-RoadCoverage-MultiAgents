from PIL import Image
import os, cv2
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from smooth_predictions_by_belnding_patches import predict_img_with_smooth_windowing
from patchify import patchify, unpatchify
from matplotlib import pyplot as plt

path = r"C:\Users\albon\Desktop"
model_path = r"C:\Users\albon\Desktop\Test\Unet-MR.h5"
file_types = [('Image', '*.jpg;*.png'), ('All files', '*')]
name = filedialog.askopenfilename(title='Select an image:', filetypes=file_types, initialdir=path)
image = cv2.imread(name, cv2.COLOR_BGR2RGB)

nr1 = int(image.shape[0] / 256.)
nc1 = int(image.shape[1] / 256.)
image = image[0:nr1*256, 0:nc1*256]

if nr1 > nc1:
    n = int( nr1/nc1 )
    image1 = cv2.resize(image, (256,256*n))
else:
    n = int( nc1/nr1 )
    image1 = cv2.resize(image, (256*n,256))

nr = int(image1.shape[0] / 256.)
nc = int(image1.shape[1] / 256.)

model = tf.keras.models.load_model(model_path, compile=False)

patch_size = 256
patches = []

patches_img = patchify(image1, (patch_size, patch_size, 3), step=patch_size)
for k in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[k,j,:,:]
        single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds
        patches.append(single_patch_img)

def recompone_images(pat, x, y):
    row = []
    backtoimg = []
    for i in range(len(pat)):
        row.append(np.array(pat[i]))  
        if (i+1) % x == 0:
            backtoimg.append(row)
            row = []
    backtoimg = np.array(backtoimg)
    img = unpatchify(backtoimg, (y*256, x*256))
    return img

#Prediction without using blending patches
mask_patches = []
for i in range(len(patches)):
    img = patches[i] / 255.0 
    p0 = model.predict(np.expand_dims(img, axis=0))[0][:, :, 0]
    p1 = model.predict(np.expand_dims(np.fliplr(img), axis=0))[0][:, :, 0]
    p1 = np.fliplr(p1)
    p2 = model.predict(np.expand_dims(np.flipud(img), axis=0))[0][:, :, 0]
    p2 = np.flipud(p2)
    p3 = model.predict(np.expand_dims(np.fliplr(np.flipud(img)), axis=0))[0][:, :, 0]
    p3 = np.fliplr(np.flipud(p3))
    thresh = 0.2
    p = (p0 + p1 + p2 + p3) / 4
    mask_patches.append(p)

prediction = recompone_images(mask_patches, nc, nr)
pred = (prediction > thresh).astype(np.uint8)

#Prediction using blending patches
input_img = image1/255.
predictions_smooth = predict_img_with_smooth_windowing(
                                                    input_img,
                                                    window_size=patch_size,
                                                    subdivisions=2,
                                                    nb_classes=1,
                                                    pred_func=(lambda img_batch_subdiv: model.predict((img_batch_subdiv)))
                                                    )

final_prediction = (predictions_smooth > thresh).astype(np.uint8)
union_prediction = (((prediction + 2*predictions_smooth[:,:,0]) / 2) > thresh).astype(np.uint8)

#plt.figure(figsize=(12, 12))
#plt.subplot(221)
#plt.title('Testing Image')
#image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
#plt.imshow(image1)
#plt.subplot(222)
#plt.title('Union Prediction')
#plt.imshow(union_prediction, cmap='gray')
#
#image_overlapped = image1.copy()
#for i in range(union_prediction.shape[0]):
#    for j in range(union_prediction.shape[1]):
#        if union_prediction[i,j] == 1:
#            image_overlapped[i,j] = 255
#
#plt.subplot(223)
#plt.title('Overlapped prediction')
#plt.imshow(image_overlapped)
#plt.show()

iimmgg = image.copy()
z = cv2.resize(union_prediction, (nc1*256, nr1*256))
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        if z[i,j] == 1:
            iimmgg[i,j] = 255

fig = plt.figure(figsize=(12,12))
fig.add_subplot(2, 1, 1)
plt.title('Original Image')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
fig.add_subplot(2, 1, 2)
plt.title('Test: Image Resize n*256 x n*256')
iimmgg = cv2.cvtColor(iimmgg, cv2.COLOR_BGR2RGB)
plt.imshow(iimmgg)

#fig.add_subplot(3, 1, 3)
#plt.title('Test: Image Resize 256 x 256')
#
#im = cv2.resize(image, (256, 256)) / 255.
#p0 = model.predict(np.expand_dims(im, axis=0))[0][:, :, 0]
#p1 = model.predict(np.expand_dims(np.fliplr(im), axis=0))[0][:, :, 0]
#p1 = np.fliplr(p1)
#p2 = model.predict(np.expand_dims(np.flipud(im), axis=0))[0][:, :, 0]
#p2 = np.flipud(p2)
#p3 = model.predict(np.expand_dims(np.fliplr(np.flipud(im)), axis=0))[0][:, :, 0]
#p3 = np.fliplr(np.flipud(p3))
#thresh = 0.2
#p = (((p0 + p1 + p2 + p3) / 4) > thresh).astype(np.uint8)
#i = cv2.resize(p, (nc1*256, nr1*256))
#io = image.copy()
#for k in range(i.shape[0]):
#    for j in range(i.shape[1]):
#        if i[k,j] == 1:
#            io[k,j] = 255
#plt.imshow(io)
#

plt.show()
