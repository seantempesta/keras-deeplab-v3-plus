from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm
from model import Deeplabv3
import coremltools
from coremltools.proto import NeuralNetwork_pb2
from matplotlib import pyplot as plt
import cv2 # used for resize. if you dont have it, use anything else
from PIL import Image

MODEL_DIR = 'models'

backbone = 'mobilenetv2'
print('Instantiating an empty Deeplabv3+ model...')
keras_model = Deeplabv3(input_shape=(512, 512, 3),
                      classes=21, backbone=backbone, weights=None)

WEIGHTS_DIR = 'weights/' + backbone
print('Loading weights from', WEIGHTS_DIR)
for layer in tqdm(keras_model.layers):
    if layer.weights:
        weights = []
        for w in layer.weights:
            weight_name = os.path.basename(w.name).replace(':0', '')
            weight_file = layer.name + '_' + weight_name + '.npy'
            weight_arr = np.load(os.path.join(WEIGHTS_DIR, weight_file))
            weights.append(weight_arr)
        layer.set_weights(weights)


# CoreML model needs to normalize the input (by converting image bits from (-1,1)), which is 
# why I'm doing the image_scale, red, green, and blue bias
print('converting...')
coreml_model = coremltools.converters.keras.convert(keras_model,input_names=['input_1'],
                         image_input_names='input_1', 
                         output_names='bilinear_upsampling_2',
                         image_scale=2/255.0,
                         red_bias=-1,
                         green_bias=-1,
                         blue_bias=-1)

coreml_model.save('DeeplabMobilenet.mlmodel')
print('model converted')

# Let's compare results!
#

# Read in a test image
img = plt.imread("imgs/image1.jpg")

# resize the image to fit the 512,512 shape
# The keras model expects the image input values to be from (-1,1), so we write a rescaled version too
w, h, _ = img.shape
ratio = 512. / np.max([w,h])
resized = cv2.resize(img,(int(ratio*h),int(ratio*w)))
rescaled = resized / 127.5 - 1.

# Padding will be necessary as the image is unlikely to be a square after being resized, but the
# neural networks will expect a 512,512 shape
pad_x = int(512 - resized.shape[0])
padded = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')
padded_rescaled = np.pad(rescaled,((0,pad_x),(0,0),(0,0)),mode='constant')

# keras results
orig_output = keras_model.predict(np.expand_dims(padded_rescaled,0))
orig_labels = np.argmax(orig_output.squeeze(),-1)
plt.imshow(orig_labels[:-pad_x])
plt.savefig("imgs/image1-keras.jpg")

# coreml results (the coreml model expects a normal image (not rescaled, but still padded)
im = Image.fromarray(padded)
coreml_output = coreml_model.predict({'input_1':im})
coreml_data = coreml_output["bilinear_upsampling_2"]
coreml_labels = np.argmax(coreml_data, 0)
plt.imshow(coreml_labels[:-pad_x])
plt.savefig("imgs/image1-coreml.jpg")
