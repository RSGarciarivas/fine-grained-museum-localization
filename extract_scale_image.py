from utils import load_dataset, make_prediction, get_calibration_mat, split_dataset
from mpi4py import MPI
import numpy as np
import pandas as pd
import cv2

# camera parameters
H_FoV = 73.7
V_FoV = 53.1

# image dimensions
W = 680
H = 490

TRAIN = './Dataset/train/'
TEST = './Dataset/test/'
JPG = '.jpg'

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    labels = load_dataset('image_labels')
    matches = pd.read_csv('matches.csv')
    feat, no_feat = load_dataset('training_set')
    
    for match in matches.match1:
        try:
            feat.pop(match)
        except KeyError:
            continue

    feat = split_dataset(feat, size)
    K = get_calibration_mat(H_FoV, V_FoV, W, H)

    matches2= []
else:
    labels = None
    matches = None
    feat, no_feat = None, None
    matches2 = None
    K = None

labels = comm.bcast(labels, root = 0)
matches = comm.bcast(matches, root = 0)
no_feat = comm.bcast(no_feat, root = 0)
K = comm.bcast(K, root = 0)


# send features to worker nodes
if rank == 0:
    for i in range(1, size):
        comm.send(feat[i], dest = i)

    feat = feat[0]
else:
    # receive data from master node
    feat = comm.recv(source = 0)

for t in matches.test:
    img = cv2.imread(TEST + t + JPG, 0)

    match2 = make_prediction(img, feat, no_feat, labels, K)
    match2 = comm.gather(match2, root = 0)

    if rank == 0:
        num_matches = [n[-1] for n in match2]
        match_idx = np.argmax(num_matches)

        matches2.append(match2[match_idx][0])

if rank == 0:
    matches['match2'] = matches2
    matches.to_csv('two_matches.csv', index = False)
