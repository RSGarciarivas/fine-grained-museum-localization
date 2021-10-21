from utils import load_dataset, get_calibration_mat, get_best_match
import numpy as np
import pandas as pd
import cv2

TRAIN = './Dataset/train/'
TEST = './Dataset/test/'
JPG = '.jpg'

# initialise camera matrix
H_FoV = 73.7
V_FoV = 53.1

W = 680
H = 490

K = get_calibration_mat(H_FoV, V_FoV, W, H)

x, y = [], []
labels = load_dataset('image_labels')
matches = pd.read_csv('two_matches.csv')

feat, no_feat = load_dataset('training_set')
no_feat = set(no_feat.keys())

# main loop for calculating scale and predicting images
for i in range(len(matches)):
    test = matches.test[i]
    match1 = matches.match1[i]
    match2 = matches.match2[i]

    # use coordinates of closest image if no features
    if match1 in no_feat:
        x1, y1 = labels[match1]
        x.append(x1)
        y.append(y1)
        continue

    # eliminate the need for searches
    temp_feat = {match2: feat[match2]}
    (x1, y1), (x2, y2) = labels[match1], labels[match2]
    
    img1 = cv2.imread(TRAIN + match1 + JPG, 0)
    test_img = cv2.imread(TEST + test + JPG, 0)

    _, good_matches, [kp_q, kp_m] = get_best_match(img1, temp_feat, no_feat)

    pts_q = np.float32([kp_q[pt.queryIdx].pt for pt in good_matches]).reshape(-1,1,2)
    pts_m = np.float32([kp_m[pt.trainIdx].pt for pt in good_matches]).reshape(-1,1,2)

    E, _ = cv2.findEssentialMat(pts_q, pts_m, K)
    try:
        _, R, t, _ = cv2.recoverPose(E, pts_q, pts_m, K)
    except:
        # sometimes not enough features to compute E
        x.append(x1)
        y.append(y1)
        continue

    scale_x = (x2 - x1) / t[0]
    scale_y = (y2 - y1) / t[1]

    temp_feat.pop(match2)
    temp_feat[match1] = feat[match1]

    _, good_matches, [kp_q, kp_m] = get_best_match(test_img, temp_feat, no_feat)

    pts_q = np.float32([kp_q[pt.queryIdx].pt for pt in good_matches]).reshape(-1,1,2)
    pts_m = np.float32([kp_m[pt.trainIdx].pt for pt in good_matches]).reshape(-1,1,2)

    _, R1, t1, _ = cv2.recoverPose(E, pts_q, pts_m, K)

    x3 = x1 + scale_x * t1[0]
    y3 = y1 + scale_y * t1[1]

    if x3 > 95 or x3 < -175:
        x.append(x1)
    else:
         x.append(x3.item())
    
    if y3 > 150 or y3 < -95:
        y.append(y1)
    else:
        y.append(y3.item())

results = pd.DataFrame()
results['id'] = matches.test
results['x'] = x
results['y'] = y

results = results.sort_values('id')
results.to_csv('scale1.csv', index = False)