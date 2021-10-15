# Necessary imports
import glob
import cv2 as cv
import numpy as np
import pickle
from matplotlib import pyplot as plt

# Builds dictionary containing SIFT features data for all images in a directory
def SIFT_feature_dataset(img_paths):
    
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
def save_dataset(name, data):
    with open(name + '.pkl', 'wb') as output_file:
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

# Get "good" matches between 2 images using its SIFT descriptors
def match_features(des_1, des_2, flann, L_ratio = 0.7):
    
    # Obtain KNN matches
    matches = flann.knnMatch(des_1, des_2, k = 2)
    
    # List for storing good matches
    good_matches = []
    
    # Get unique, good enough matches
    for m, n in matches:
        
        # Check if it's a good enough match
        if m.distance < L_ratio * n.distance:
            
            # Check if match is already present in good_matches
            if m.trainIdx in [match.trainIdx for match in good_matches]:
                # Get location of that match in list
                index = [match.trainIdx for match in good_matches].index(m.trainIdx)
                
                # Substitute old match if this match is better
                if m.distance < good_matches[index].distance:
                    good_matches[index] = m
                        
            # If match is not yet present, add it
            else:
                good_matches.append(m)
    
    return good_matches


def get_best_match(img, dataset):
    
    # Get SIFT features for image of interest
    sift = cv.SIFT_create()
    kp_query, des_query = sift.detectAndCompute(img, None)
    
    # Dictionary for storing best match so far
    best_match = {'name': '', 'matches': 0, 'good_matches': [], 'keypoints': []}
    
    # Set up Flann based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    # Go through each image in the dataset
    for i, img_name in enumerate(dataset):
        
        print(f'Image {i+1} of {len(dataset)}. Best match so far: {best_match["matches"]} points.')
        
        # Load its SIFT features
        kp, des = get_img_features(img_name, dataset)
        
        # Check both images have descriptors
        if (des is not None) and (des_query is not None):
            # Check each image has more than 2 descriptors
            if (len(des) > 2) and (len(des_query) > 2):
                
                # Get matching features
                good_matches = match_features(des_query, des, flann)
                
                # Update best match so far if better match found
                if len(good_matches) > best_match['matches']:
                    best_match['name'] = img_name
                    best_match['matches'] = len(good_matches)
                    best_match['good_matches'] = good_matches
                    best_match['keypoints'] = kp
    
    return best_match['name'], best_match['good_matches'], [kp_query, best_match['keypoints']]


