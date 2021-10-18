# Necessary imports
import glob
import cv2 as cv
import numpy as np
import pickle
from matplotlib import pyplot as plt
import csv

# Builds entry of SIFT features for image with enough descriptors
def add_img_features(kp, des):
    
    img_features = {}
    keypoints = []
    
    for keypoint in kp:
        # Extract keypoint info in a way that can be stored in a file
        keypoint = {'pt': keypoint.pt, 'size': keypoint.size,
                    'angle': keypoint.angle}
        
        keypoints.append(keypoint)
    
    # Append them in dictionary
    img_features = {'keypoints': keypoints, 'descriptors': des}
    
    return img_features
    
# Builds entry for image with insufficient descriptors
def add_no_features_img(img):
    
    # Returns the mean pixel value of the image
    return np.mean(img)

# From all images in a directory, builds dictionary containing SIFT features
# data for all images with more than the minimum descriptors and dictionary 
# containing images with less than the minimum number of descriptors 
def build_dataset(img_paths, N = 5):
    
    print(f'{len(img_paths)} images to analyse.')

    # Dictionary for storing features for each image
    SIFT_features = {}
    no_features = {}
    
    # Work on each image
    for i, img_path in enumerate(img_paths):
          
        print(f'Working on image {i + 1} of {len(img_paths)}')
        
        # Read image
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        
        # Obtain SIFT keypoints and descriptors
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        
        # Add entry to suitable dictionary
        if len(kp) >= 5:
            
            # Append features in dictionary
            SIFT_features[img_path[16:-4]] = add_img_features(kp, des)
        
        else:
            
            # Append image into images with not enough features
            no_features[img_path[16:-4]] = add_no_features_img(img)
    
    return SIFT_features, no_features

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

# Find best match between images with sufficient SIFT descriptors
def find_best_SIFT_match(kp_q, des_q, dataset, kp_threshold = 120, kp_ratio = 0.7):
    
    # Dictionary for storing best match so far
    best_match = {'name': '', 'matches': 0, 'good_matches': [], 'keypoints': []}
    
    # Set up Flann based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    # Go through each image in the dataset
    for i, img_name in enumerate(dataset):
        
        # print(f'Image {i+1} of {len(dataset)}. Best match so far: {best_match["matches"]} points.')
        
        # Load its SIFT features
        kp_m, des_m = get_img_features(img_name, dataset)
                
        # Get matching features
        good_matches = match_features(des_q, des_m, flann)
        
        # Update best match so far if better match found
        if len(good_matches) > best_match['matches']:
            best_match['name'] = img_name
            best_match['matches'] = len(good_matches)
            best_match['good_matches'] = good_matches
            best_match['keypoints'] = kp_m
        
        # Break loop early if a good match has been found
        if len(good_matches) >= kp_ratio*len(kp_q) or (len(good_matches) >= kp_threshold):
            print('here')
            break
    
    return best_match['name'], best_match['good_matches'], [kp_q, best_match['keypoints']]

# Find best match between images that don't have enough SIFT descriptors
def find_best_no_feature_match(img, dataset):
    
    # Dictionary for storing best match so far
    best_match = {'name': '', 'mean difference': float('inf')}
    
    # Get mean pixel value for the image
    mpv_q = np.mean(img)
    
    # Compare to every image in dataset
    for img_name, mpv_m in dataset.items():
        err = abs(mpv_q - mpv_m)
        
        # Update best match so far if better match found
        if err < best_match['mean difference']:
            best_match['name'] = img_name
            best_match['mean difference'] = err
    
    return best_match['name'], None, [None, None]

# Get an image's best match by SIFT features from the given dataset
def get_best_match(img, feature_dataset, no_feature_dataset):
    
    # Get SIFT features for image of interest
    sift = cv.SIFT_create()
    kp_q, des_q = sift.detectAndCompute(img, None)
    
    # Check if image has enough SIFT descriptors for feature matching
    if len(kp_q) > 5:
        
        # Find best match according to its SIFT descriptors
        # print('Matching according to SIFT descriptors...')
        match_name, good_matches, [kp_q, kp_m] = find_best_SIFT_match(kp_q, des_q, 
                                                                      feature_dataset)
        
    else:
        
        # Find best match without features
        # print('Matching according to similarity...')
        match_name, good_matches, [kp_q, kp_m] = find_best_no_feature_match(img, no_feature_dataset)
    
    return match_name, good_matches, [kp_q, kp_m]

# Get calibration matrix given a camera's fields of view
# in both axes and the dataset images' dimensions
def get_calibration_mat(H_FoV, V_FoV, img_W, img_H):
    # Horizontal and vertical focal lengths
    fx = (img_W / 2) / np.tan((H_FoV*np.pi/180) / 2)
    fy = (img_H / 2) / np.tan((V_FoV*np.pi/180) / 2)
    
    K = np.array([[fy, 0, img_W/2], [0, fx, img_H/2], [0, 0, 1]])
    
    return K







def get_location(img_query, dataset, labels, K):
    
    # Get best match from the dataset
    img_match_name, good_matches, [kp_query, kp_match] = get_best_match(img_query, dataset)
    
    # Get coordinates of best match
    Xm = np.array(labels[img_match_name])
    
    # Get matching points for both images
    pts_q = np.float32([kp_query[pt.queryIdx].pt for pt in good_matches]).reshape(-1,1,2)
    pts_m = np.float32([kp_match[pt.trainIdx].pt for pt in good_matches]).reshape(-1,1,2)
    
    # Find the essential matrix for this pair of images
    E, _ = cv.findEssentialMat(pts_q, pts_m, K)
    
    # Recover relative rotation and translation from E
    _, R, t, _ = cv.recoverPose(E, pts_q, pts_m, K)
    
    
    
    
    
    
    return True





# Predict the location of an image based on the given datasets
def make_prediction(img_q, feature_dataset, no_feature_dataset, labels, K):
    
    # Get best match from the datasets
    match_name, good_matches, [kp_q, kp_m] = get_best_match(img_q, feature_dataset,
                                                            no_feature_dataset)
    
    # return its coordinates
    return match_name, labels[match_name]

# Predicts coordinates for all images in given directory
def predict_and_write_output(output_file, img_paths, feature_dataset, no_feature_dataset, labels, K):
    
    # Open csv file for writing results
    with open(output_file + '.csv', 'w') as out_csv:
        writer = csv.writer(out_csv, delimiter = ',')
        
        # Write headers
        writer.writerow(['id', 'x', 'y'])
        
        # Analyse each image in directory
        for i, img_path in enumerate(img_paths):
            
            print(f'Image {i+1} of {len(img_paths)}')
            img_name = img_path[15:-4]
            # Read image
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            
            # Predict coordinates
            _, Xq = make_prediction(img, feature_dataset, no_feature_dataset, labels, K)
            # Write result
            writer.writerow([img_name, Xq[0], Xq[1]])
    

