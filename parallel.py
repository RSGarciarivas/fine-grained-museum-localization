from utils import load_dataset, make_prediction, get_calibration_mat, split_dataset
from mpi4py import MPI
import numpy as np
import glob
import cv2
import csv



# camera parameters
H_FoV = 73.7
V_FoV = 53.1

# image dimensions
W = 680
H = 490

# initialise MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# load in data via master node
if rank == 0:
    labels = load_dataset('image_labels')
    feat, no_feat = load_dataset('training_set')
    feat = split_dataset(feat, size)
    test_names = glob.glob('./Dataset/test/*')
    K = get_calibration_mat(H_FoV, V_FoV, W, H)

    # for results
    match_name = []
    coords = []
else:
    labels = None
    feat, no_feat = None, None
    test_names = None
    K = None

# send data to worker nodes
labels = comm.bcast(labels, root = 0)
no_feat = comm.bcast(no_feat, root = 0)
test_names = comm.bcast(test_names, root = 0)
K = comm.bcast(K, root = 0)

# send features to worker nodes
if rank == 0:
    for i in range(1, size):
        comm.send(feat[i], dest = i)

    feat = feat[0]
else:
    # receive data from master node
    feat = comm.recv(source = 0)

# loop through names for each subset of features (all nodes)
for t in test_names:
    img = cv2.imread(t, 0)

    match = make_prediction(img, feat, no_feat, labels, K)
    match = comm.gather(match, root = 0)

    if rank == 0:
        num_matches = [n[-1] for n in match]
        match_idx = np.argmax(num_matches)

        match_name.append(match[match_idx][0])
        coords.append(match[match_idx][1])

# write results to csv
if rank == 0:
    if len(match_name) != len(coords):
        print('Error')
    else:
        test_n = len(match_name)

        with open('results.csv', 'w') as f:
            writer = csv.writer(f, delimiter = ',')
            writer.writerow(['id','x','y'])

            for i in range(test_n):
                writer.writerow([match_name[i], coords[i][0], coords[i][1]])