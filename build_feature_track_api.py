import cv2
import numpy as np
from scipy.spatial import cKDTree

import torch
import cv2

import argparse
import fnmatch

from torch.utils.data.dataset import Dataset
import os

import sys
from tqdm import tqdm

from utils import *

sys.path.append('./SuperGlueMatching')
from models.utils import read_image
from models.superpoint import SuperPoint


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images

    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (m, 2)
        The matched keypoint locations in image 1
    x2 : ndarray of shape (m, 2)
        The matched keypoint locations in image 2
    ind1 : ndarray of shape (m,)
        The indices of x1 in loc1
    """
    tree1 = cKDTree(des1)
    dist1, idx1 = tree1.query(des2, k=2, workers=-1)
    mask1 = (dist1[:, 0] / dist1[:, 1]) < 0.9

    tree2 = cKDTree(des2)
    dist2, idx2 = tree2.query(des1, k=2, workers=-1)
    mask2 = (dist2[:, 0] / dist2[:, 1]) < 0.9

    mask = mask2 * mask1[idx2[:, 0]] * (idx1[idx2[:, 0], 0] == np.arange(des1.shape[0]))
    ind1 = np.flatnonzero(mask)

    x1 = loc1[mask, :]
    x2 = loc2[idx2[mask, 0], :]

    _, x10_indices = np.unique(x1[:, 0], return_index=True)
    _, x11_indices = np.unique(x1[:, 1], return_index=True)
    _, x20_indices = np.unique(x2[:, 0], return_index=True)
    _, x21_indices = np.unique(x2[:, 1], return_index=True)
    mask_unique = np.intersect1d(x10_indices, x11_indices)
    mask_unique = np.intersect1d(mask_unique, x20_indices)
    mask_unique = np.intersect1d(mask_unique, x21_indices)

    x1 = x1[mask_unique]
    x2 = x2[mask_unique]
    ind1 = ind1[mask_unique]

    return x1, x2, ind1


def EstimateE(x1, x2):
    """
    Estimate the essential matrix, which is a rank 2 matrix with singular values
    (1, 1, 0)

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    """
    # A is in shape (n, 9)
    A = np.stack([
        x1[:, 0] * x2[:, 0],
        x1[:, 1] * x2[:, 0],
        x2[:, 0],
        x1[:, 0] * x2[:, 1],
        x1[:, 1] * x2[:, 1],
        x2[:, 1],
        x1[:, 0],
        x1[:, 1],
        np.ones_like(x1[:, 0])
    ], axis=1)

    U, S, Vh = np.linalg.svd(A)
    f = Vh[-1, :]
    E = np.reshape(f, (3, 3))

    U, S, Vh = np.linalg.svd(E)
    S = np.eye(3)
    S[2, 2] = 0
    E = U @ S @ Vh

    return E


def EstimateE_RANSAC(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the essential matrix robustly using RANSAC

    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Set of correspondences in the first image
    x2 : ndarray of shape (n, 2)
        Set of correspondences in the second image
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    E : ndarray of shape (3, 3)
        The essential matrix
    inlier : ndarray of shape (k,)
        The inlier indices
    """
    x1_h = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2_h = np.hstack([x2, np.ones((x2.shape[0], 1))])

    max_inlier = 0
    E = np.eye(3)
    inlier = None
    for i in range(ransac_n_iter):
        rand_idx = np.random.choice(x1.shape[0], size=8, replace=False)
        x1_r = x1[rand_idx, :2]
        x2_r = x2[rand_idx, :2]
        E_r = EstimateE(x1_r, x2_r)

        e = np.abs((x2_h @ E_r * x1_h).sum(axis=1)) / np.linalg.norm(E_r[:2, :] @ x1_h.T, axis=0)
        inlier_mask = e < ransac_thr
        if inlier_mask.sum() > max_inlier:
            E = E_r
            inlier = np.flatnonzero(inlier_mask)
            max_inlier = inlier_mask.sum()

    return E, inlier


def BuildFeatureTrack(Im, K):
    """
    Build feature track

    Parameters
    ----------
    Im : ndarray of shape (N, H, W, 3)
        Set of N images with height H and width W
    K : ndarray of shape (3, 3)
        Intrinsic parameters

    Returns
    -------
    track : ndarray of shape (N, F, 2)
        The feature tensor, where F is the number of total features
    """
    # Extract SIFT descriptors
    if str.startswith(cv2.__version__, '3'):
        sift = cv2.xfeatures2d.SIFT_create()
    elif str.startswith(cv2.__version__, '4'):
        sift = cv2.SIFT_create()
    num_images = Im.shape[0]
    loc_list = []
    des_list = []
    for i in range(num_images):
        kp, des = sift.detectAndCompute(cv2.cvtColor(Im[i, :, :, :], cv2.COLOR_RGB2GRAY), None)
        loc = np.asarray([kp[j].pt for j in range(len(kp))])
        loc_list.append(loc)
        des_list.append(des)

    ransac_n_iter = 200
    ransac_thr = 0.003
    track = None
    for i in range(num_images - 1):
        num_points = loc_list[i].shape[0]
        # Initialize track_i as -1
        track_i = -np.ones((num_images, num_points, 2))
        mask = np.zeros((num_points,), dtype=bool)
        for j in range(i + 1, num_images):
            # Match features between the i-th and j-th images
            x1, x2, ind1 = MatchSIFT(loc_list[i], des_list[i], loc_list[j], des_list[j])
            # Normalize coordinate by multiplying the inverse of intrinsics
            x1_n = np.hstack([x1, np.ones((x1.shape[0], 1))]) @ np.linalg.inv(K).T
            x1_n = x1_n[:, :2]
            x2_n = np.hstack([x2, np.ones((x2.shape[0], 1))]) @ np.linalg.inv(K).T
            x2_n = x2_n[:, :2]

            # Find inliner matches using essential matrix
            E, inlier = EstimateE_RANSAC(x1_n, x2_n, ransac_n_iter, ransac_thr)
            print('Matching: {} <-> {}: {}'.format(i + 1, j + 1, inlier.size))

            # Update track_i using the inlier matches
            track_i[i, ind1[inlier], :] = x1_n[inlier]
            track_i[j, ind1[inlier], :] = x2_n[inlier]
            mask[ind1[inlier]] = True

        # Remove features in track_i that have not been matched for i+1, ..., N
        track_i = track_i[:, mask, :]
        # Append track_i to track
        if track is None:
            track = track_i
        else:
            track = np.concatenate([track, track_i], axis=1)

    return track


class AzureKinectMatching(Dataset):
    def __init__(self, keypoint_list, descriptor_list, keyframes_list, K, max_numpoints=650):
        super(AzureKinectMatching, self).__init__()
        self.num_images = len(keyframes_list)
        self.kpts = keypoint_list
        self.desc = descriptor_list
        self.keyframes_list = keyframes_list
        self.K = K
        self.num_point = max_numpoints

    def __getitem__(self, index):
        # Initialize track_i as -1
        track_i = -np.ones((len(self.keyframes_list), self.num_point, 2))
        mask = np.zeros((self.num_point,), dtype=bool)
        # for j in range(i + 1, min(len(keyframes_list), i + 100)):
        for j in range(index + 1, min(index + 200, len(self.keyframes_list))):
            # Match features between the i-th and j-th images
            x1, x2, ind1 = MatchSIFT(loc_list[self.keyframes_list[index]],
                                     des_list[self.keyframes_list[index]],
                                     loc_list[self.keyframes_list[j]],
                                     des_list[self.keyframes_list[j]])

            # Normalize coordinate by multiplying the inverse of intrinsics
            x1_n = np.hstack([x1, np.ones((x1.shape[0], 1))]) @ np.linalg.inv(self.K).T
            x1_n = x1_n[:, :2]
            x2_n = np.hstack([x2, np.ones((x2.shape[0], 1))]) @ np.linalg.inv(self.K).T
            x2_n = x2_n[:, :2]

            # Find inliner matches using essential matrix
            if x1_n.shape[0] > 40:  # Minimum requirement for RANSAC to work
                E, inlier = EstimateE_RANSAC(x1_n, x2_n, ransac_n_iter, ransac_thr)
                # print('Matching: {} <-> {}: {}'.format(i, j, inlier.size))
                if inlier.size > 30:
                    track_i[index, ind1[inlier], :] = x1_n[inlier]
                    track_i[j, ind1[inlier], :] = x2_n[inlier]
                    mask[ind1[inlier]] = True

        output = {'track': torch.tensor(track_i),
                  'mask': torch.tensor(mask)}

        return output

    def __len__(self):
        return self.num_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--extract_descriptor', action='store_true')
    parser.add_argument(
        '--ego_dataset_folder', type=str,
        help='Ego dataset folder')

    opt = parser.parse_args()
    print(opt)

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    superpoint = SuperPoint(config.get('superpoint', {})).cuda()
    superpoint = superpoint.eval()


    ROOT_DIR = opt.ego_dataset_folder
    OUTPUT_DIR = 'track'


    if not os.path.exists(os.path.join(ROOT_DIR, OUTPUT_DIR)):
        os.mkdir(os.path.join(ROOT_DIR, OUTPUT_DIR))

    # EXTRACT KPTS AND DESCRIPTORS, RUN ONCE!
    if opt.extract_descriptor:
        loc_list = []
        des_list = []
        image_subsample = 1
        max_numpoint = -1


        all_images = np.arange(0, len(fnmatch.filter(os.listdir(os.path.join(ROOT_DIR, 'color')), '*.jpg')), step=1)

        for index in all_images: # range(original_img_idx_start, original_img_idx_end, image_subsample):
            color_info = os.path.join(ROOT_DIR, 'color/color_%07d.jpg' % index)
            _, gray_tensor, scale = read_image(color_info, 'cpu', resize=[640, 480], rotation=0, resize_float=False)
            gray_tensor = gray_tensor.reshape(1, 1, 480, 640)
            input_batch = {'image': gray_tensor.cuda()}

            with torch.no_grad():
                output = superpoint(input_batch)

            output_np = {k: [output[k][i].detach().cpu().numpy() for i in range(len(output[k]))] for k in output}

            # kpts = [cv2.KeyPoint(output_np['keypoints'][0][k, 0] * 3.0,
            #                      output_np['keypoints'][0][k, 1] * 2.25, 50)
            #                      for k in range(output_np['keypoints'][0].shape[0])]

            des = np.asarray(output_np['descriptors'][0])
            loc = np.asarray(
                [output_np['keypoints'][0][j] * scale for j in range(output_np['keypoints'][0].shape[0])])

            loc_list.append(loc)
            des_list.append(des.transpose())

            if max_numpoint < output_np['keypoints'][0].shape[0]:
                max_numpoint = output_np['keypoints'][0].shape[0]
            print('image ', index, ' feature size ', loc.shape, ' descriptor size ', des.shape, ' max numpoint ',
                  max_numpoint)

        np.savez('%s/keypoints.npz' % os.path.join(ROOT_DIR, OUTPUT_DIR), loc_list)
        np.savez('%s/descriptors.npz' % os.path.join(ROOT_DIR, OUTPUT_DIR), des_list)

    loc_list = np.load(os.path.join(ROOT_DIR, OUTPUT_DIR, 'keypoints.npz'), allow_pickle=True)['arr_0']
    des_list = np.load(os.path.join(ROOT_DIR, OUTPUT_DIR, 'descriptors.npz'), allow_pickle=True)['arr_0']


    ransac_n_iter = 200
    ransac_thr = 0.005
    track = None
    num_images = len(loc_list)
    print('num image: ', num_images)
    K = np.loadtxt('%s/intrinsics.txt' % ROOT_DIR)

    original_image_id_list = np.arange(0, len(fnmatch.filter(os.listdir(opt.ego_dataset_folder + '/color/'), '*.jpg')), step=1)
    keyframes_list = np.arange(0, original_image_id_list.shape[0])


    for i in tqdm(range(len(keyframes_list) - 1)):
        num_points = loc_list[keyframes_list[i]].shape[0]

        # Initialize track_i as -1
        track_i = -np.ones((len(keyframes_list), num_points, 2))
        mask = np.zeros((num_points,), dtype=bool)
        for j in range(i + 1, min(len(keyframes_list), i + 200)): # len(keyframes_list)):  #min(len(keyframes_list), i + 200)):
            # Match features between the i-th and j-th images
            x1, x2, ind1 = MatchSIFT(loc_list[keyframes_list[i]],
                                     des_list[keyframes_list[i]],
                                     loc_list[keyframes_list[j]],
                                     des_list[keyframes_list[j]])

            # Normalize coordinate by multiplying the inverse of intrinsics
            x1_n = np.hstack([x1, np.ones((x1.shape[0], 1))]) @ np.linalg.inv(K).T
            x1_n = x1_n[:, :2]
            x2_n = np.hstack([x2, np.ones((x2.shape[0], 1))]) @ np.linalg.inv(K).T
            x2_n = x2_n[:, :2]

            # Find inliner matches using essential matrix
            if x1_n.shape[0] > 40:  # Minimum requirement for RANSAC to work
                E, inlier = EstimateE_RANSAC(x1_n, x2_n, ransac_n_iter, ransac_thr)
                # print('Matching: {} <-> {}: {}'.format(i, j, inlier.size))
                if inlier.size > 15:
                    track_i[i, ind1[inlier], :] = x1_n[inlier]
                    track_i[j, ind1[inlier], :] = x2_n[inlier]
                    mask[ind1[inlier]] = True

        # Remove features in track_i that have not been matched for i+1, ..., N
        track_i = track_i[:, mask, :]
        # Append track_i to track
        if track is None:
            track = track_i
        else:
            # simple harshing
            ti_new = track_i[i, :, 0] * np.exp(track_i[i, :, 1])
            ti_old = track[i, :, 0] * np.exp(track[i, :, 1])
            existing_track_ids = np.in1d(ti_new, ti_old)

            ti_old_sorted_idx = np.argsort(ti_old)
            ti_new_pos = np.searchsorted(ti_old[ti_old_sorted_idx], ti_new[existing_track_ids])
            existing_track_ids_in_track = ti_old_sorted_idx[ti_new_pos]

            track[(i + 1):num_images, existing_track_ids_in_track] = track_i[(i + 1):num_images, existing_track_ids]

            new_track_ids = ~existing_track_ids
            track = np.concatenate([track, track_i[:, new_track_ids]], axis=1)
            # print('Found %d existing matches ' % existing_track_ids_in_track.shape[0])
            # print('Found %d new matches ' % np.sum(new_track_ids))
            # print('total number of feature so far: ', track.shape[1])

        if i % 100 == 0:
            np.save('%s/track.npy' % os.path.join(ROOT_DIR, OUTPUT_DIR), track)
            np.save('%s/original_image_id.npy' % os.path.join(ROOT_DIR, OUTPUT_DIR), original_image_id_list)

    # np.save('track_subsample_10.npy', track)
    # np.save('original_image_subsample_10.npy', original_image_id_list)
    print('total number of feature: ', track.shape[1])