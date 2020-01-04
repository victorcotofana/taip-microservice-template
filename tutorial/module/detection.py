

import pickle
import numpy as np

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey
# from sklearn.decomposition import PCA
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from django.conf import settings
import io
from io import BytesIO
import os
import base64

print('## ', settings.MODEL_PATH)
MODEL_PATH = settings.MODEL_PATH
HOG_BLOCK_NORM = 'L2-Hys'
SVM_PXS_PER_CELL = (16, 16)


def image_to_array(file_path, base64img=False):
    if base64img:
        img = Image.open(BytesIO(base64.b64decode(file_path)))
    else:
        img = Image.open(file_path)
    return np.array(img)


def create_features(img_array):
    # flatten three channel color image
    color_features = img_array.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img_array)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, visualize=False, block_norm=HOG_BLOCK_NORM, pixels_per_cell=SVM_PXS_PER_CELL)

    # combine color and hog features into a single array
    flat_features = np.hstack((color_features, hog_features))

    return flat_features

def load_and_predict(img_path):
    # load model again
    with open(MODEL_PATH, 'rb') as fid:
        loaded_svm = pickle.load(fid)

    img_features = [create_features(image_to_array(img_path, base64img=True))]
    # img_features = [create_features(image_to_array(img_path))]

    prediction = loaded_svm.predict(img_features)

    if prediction == [1.]:
        return 1
    else:
        return 0