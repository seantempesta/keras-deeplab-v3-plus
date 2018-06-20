from __future__ import print_function

import os
import numpy as np
from tqdm import tqdm
from model import Deeplabv3
import coremltools
from coremltools.proto import NeuralNetwork_pb2

MODEL_DIR = 'models'

backbone = 'mobilenetv2'
print('Instantiating an empty Deeplabv3+ model...')
model = Deeplabv3(input_shape=(512, 512, 3),
                      classes=21, backbone=backbone, weights=None)

def convert_bilinear_upsampling(layer):
    params = NeuralNetwork_pb2.CustomLayerParams()

    # The name of the Swift or Obj-C class that implements this layer.
    params.className = "BilinearUpsampling"

    # The desciption is shown in Xcode's mlmodel viewer.
    params.description = ""

    # Set configuration parameters
    return params

print('converting...')
coreml_model = coremltools.converters.keras.convert(model,input_names=['input_1'],
                         image_input_names='input_1', 
                         output_names='bilinear_upsampling_2', 
                         add_custom_layers=True,
                         custom_conversion_functions={ "BilinearUpsampling": convert_bilinear_upsampling })
coreml_model.save('DeeplabMobilenet.mlmodel')
print('model converted')
