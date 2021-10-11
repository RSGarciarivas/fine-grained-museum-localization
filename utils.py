# Necessary imports
import glob
import cv2 as cv
import numpy as np
import pickle
from matplotlib import pyplot as plt

# Builds dictionary containing SIFT features data for all images in a directory
def SIFT_feature_dataset(path):
    
    # Get list of all images
    img_paths = glob.glob(path)
    
    print(f'{len(img_paths)} images to analyse.')

    # Dictionary for storing features for each image
    SIFT_features = {}
    
    # Work on each image
    for i, img_path in enumerate(img_paths):
          
        print(f'Working on image {i + 1} of {len(img_paths)}')
        
        # Read image
        img = cv.imread(img_path)
        
        # Obtain SIFT keypoints and descriptors
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        
        # List for storing all keypoints' data
        keypoints = []
        
        for keypoint in kp:
            # Extract keypoint info in a way that can be stored in a file
            keypoint = {'pt': keypoint.pt, 'size': keypoint.size,
                        'angle': keypoint.angle}
            
            keypoints.append(keypoint)
        
        # Append them in dictionary
        SIFT_features[img_path[16:-4]] = {'keypoints': keypoints,
                                          'descriptors': des}
    
    return SIFT_features

# Saves data dictionary into a .pkl file
def save_dataset(data):
    with open(data + '.pkl', 'wb') as output_file:
        pickle.dump(data, output_file)

# Loads data from .pkl file
def load_dataset(filename):
    with open(filename + '.pkl', 'rb') as input_file:
        data = pickle.load(input_file)
        
        return data

# Function for easily loading features from the feature dataset
def get_img_features(img_name, dataset):
    
    # List for storing keypoint objects
    kp = []
    
    # Extract data for each keypoint from the image
    for keypoint in dataset[img_name]['keypoints']:
        point = keypoint['pt']
        size = keypoint['size']
        angle = keypoint['angle']
        
        # Build keypoint object
        feature = cv.KeyPoint(x = point[0], y = point[1], 
                              _size = size, _angle = angle)
        
        # Append to keypoint list
        kp.append(feature)
    
    # Load SIFT descriptors
    des = dataset[img_name]['descriptors']
    
    return kp, des


#def get_best_match()

# FUNCTION FOR MATCHING FEATURES WELL!!!