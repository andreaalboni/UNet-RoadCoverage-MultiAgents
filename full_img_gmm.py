#!/usr/bin/env python3
import rospy
from gmm_msgs.msg import GMM, Gaussian
from geometry_msgs.msg import Point

import time, cv2, math, os
import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from matplotlib import cm

from PIL import Image
import tensorflow as tf
from smooth_predictions_by_belnding_patches import predict_img_with_smooth_windowing
from patchify import patchify, unpatchify

my_dpi = 96
COMPONENTS_NUM = 15

def get_real_mt(DIM_MT_IMG, image_path):
    REAL_DIM_MT_IMG = []
    image = cv2.imread(image_path, 0)
    nr1 = int(image.shape[0] / 256.)
    nc1 = int(image.shape[1] / 256.)
    image_shaped = image[0:nr1*256, 0:nc1*256]
    REAL_IMG_SIZE = image_shaped.shape
    IMG_SIZE = image.shape

    REAL_DIM_MT_IMG.append( int((REAL_IMG_SIZE[1] * DIM_MT_IMG[0]) / IMG_SIZE[1]) )
    REAL_DIM_MT_IMG.append( int((REAL_IMG_SIZE[0] * DIM_MT_IMG[1]) / IMG_SIZE[0]) )

    return REAL_DIM_MT_IMG

def distmt(x, y, img_size, mt):
    dist_x = (x*mt[0])/img_size[1]
    dist_y = (y*mt[1])/img_size[0]
    return [math.sqrt(dist_x**2+dist_y**2), dist_x, dist_y]


# X, Y : meshgrid
def multigauss_pdf(X, Y, means, covariances, weights):
    # Flatten the meshgrid coordinates
    points = np.column_stack([X.flatten(), Y.flatten()])

    # Number of components in the mixture model
    num_components = len(means)

    # Initialize the probabilities
    probabilities = np.zeros_like(X)

    # Calculate the probability for each component
    for i in range(num_components):
        mean = means[i]
        covariance = covariances[i]
        weight = weights[i]

        # Calculate the multivariate Gaussian probability
        exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
        coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
        component_prob = coefficient * np.exp(exponent)

        # Add the component probability weighted by its weight
        probabilities += weight * component_prob.reshape(X.shape)

    return probabilities


def gmm_model(image, DIM_MT_IMG):
    xp, yp = [], []
    IMG_SIZE = image.shape

    for i in range(IMG_SIZE[0]):
        for j in range(IMG_SIZE[1]):
            if image[i,j] == 255: 
                #xp.append(j)
                #yp.append(IMG_SIZE[1]-i)

                # Da pos pixel a mt
                k,x,y = distmt(j, i, IMG_SIZE, DIM_MT_IMG)
                xp.append(x)
                yp.append(DIM_MT_IMG[1]-y)

    GMModel = GaussianMixture(n_components=COMPONENTS_NUM, covariance_type='full', max_iter=1000)
    GMModel.fit(np.column_stack((xp, yp)))
    # calculate BIC
    # bic = GMModel.bic(np.column_stack((xp, yp)))

    # get means and covariances
    means = GMModel.means_
    covariances = GMModel.covariances_
    mix = GMModel.weights_
    #print("Means: {}".format(means))
    #print("Coveriances: {}".format(covariances))
    #print("Mixture proportions: {}".format(mix))

    return means, covariances, mix


def create_msg(mns, cvs, mix):
    gmm_msg = GMM()
    for i in range(len(mns)):
        g = Gaussian()
        mean_pt = Point()

        mean_pt.x = mns[i][0]
        mean_pt.y = mns[i][1]
        mean_pt.z = 0.0
        g.mean_point = mean_pt
        for j in range(len(cvs[i])):
            g.covariance.append(cvs[i][j][0])
            g.covariance.append(cvs[i][j][1])

        gmm_msg.gaussians.append(g)
        gmm_msg.weights.append(mix[i])

    return gmm_msg


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


def predict(image_path, model_path):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)

    nr1 = int(image.shape[0] / 256.)
    nc1 = int(image.shape[1] / 256.)
    image = image[0:nr1*256, 0:nc1*256]

    #foto minori a 1024x1024 processate senza resize
    if nr1 > 4 or nc1 > 4:
        if nr1 > nc1:
            n = int( nr1/nc1 )
            image1 = cv2.resize(image, (256,256*n))
        else:
            n = int( nc1/nr1 )
            image1 = cv2.resize(image, (256*n,256))
        nr = int(image1.shape[0] / 256.)
        nc = int(image1.shape[1] / 256.)
    else:
        image1 = image

    model = tf.keras.models.load_model(model_path, compile=False)

    patch_size = 256
    patches = []

    patches_img = patchify(image1, (patch_size, patch_size, 3), step=patch_size)
    for k in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[k,j,:,:]
            single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds
            patches.append(single_patch_img)

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

    if nr1 > 4 or nc1 > 4:
        prediction = recompone_images(mask_patches, nc, nr)
    else:
        prediction = recompone_images(mask_patches, nc1, nr1)
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
    return union_prediction*255

if __name__ == '__main__':
    rospy.init_node("gmm_node")
    rospy.loginfo("Node has been started")

    pub = rospy.Publisher("/gaussian_mixture_model", GMM, queue_size=10)

    #image must be RGB
    #image must be RGB
    #Env1 - Reggio 
    #image_path = '/home/ubuntu/env.png'

    #Env2 - Roma
    image_path = '/home/ubuntu/env2.png'

    model_path = '/home/ubuntu/RoadExtractionModel.h5'
    prediction = predict(image_path, model_path)

    #Env1 - Reggio 
    #DIM_MT_IMG = [280, 157]

    #Env2 - Roma
    DIM_MT_IMG = [440, 266]

    DIM_MT_IMG = get_real_mt(DIM_MT_IMG, image_path)
    print(DIM_MT_IMG)

    #Show prediction:
    #plt.imshow(prediction, cmap='gray')
    #plt.show()

    mns, cov, mix = gmm_model(prediction, DIM_MT_IMG)

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        #creating a message
        msg = create_msg(mns, cov, mix)
        #------------------

        #sending the message
        pub.publish(msg)
        #------------------

        rate.sleep()