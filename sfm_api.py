from PIL import Image
from reconstruction import Triangulation_RANSAC
from reconstruction import Triangulation_LS
import argparse
import copy

from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from utils import *
import os
import numpy as np
import cv2

def convert_2d_to_3d(u, v, z, K):
    v0 = K[1][2]
    u0 = K[0][2]
    fy = K[1][1]
    fx = K[0][0]
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy
    return x, y, z


def PnP(x1s, x2s, m1_ids, K1, K2, DATABASE_IMAGE_FOLDER):

    # extract 3d pts
    pts3d_all = None
    f2d = []  # keep only feature points with depth in the current frame
    for k in range(len(x1s)):
        d1 = Image.open(os.path.join(DATABASE_IMAGE_FOLDER, 'depth/depth_%06d.png' % m1_ids[k]))
        d1 = np.asarray(d1).astype(np.float32) / 1000.0
        x1 = np.array(x1s[k])
        x2 = np.array(x2s[k])
        T1 = np.loadtxt(os.path.join(DATABASE_IMAGE_FOLDER, 'pose/pose_%06d.txt' % m1_ids[k]))

        f3d = []
        for i, pt2d in enumerate(x1):
            u, v = pt2d[0], pt2d[1]
            z = d1[int(v), int(u)]
            if z > 0:
                xyz_curr = convert_2d_to_3d(u, v, z, K1)
                f3d.append(xyz_curr)
                f2d.append(x2[i, :])
        f3d = (T1[:3, :3] @ np.array(f3d).transpose() + T1[:3, 3:]).transpose()
        if pts3d_all is None:
            pts3d_all = f3d
        else:
            pts3d_all = np.concatenate((pts3d_all, f3d), axis=0)

    # the minimal number of points accepted by solvePnP is 4:
    f3d = np.expand_dims(pts3d_all.astype(np.float32), axis=1)

    f2d = np.expand_dims(
        np.array(f2d).astype(np.float32), axis=1)


    ret = cv2.solvePnPRansac(f3d,
                             f2d,
                             K2,
                             distCoeffs=None,
                             flags=cv2.SOLVEPNP_EPNP)
    success = ret[0]
    rotation_vector = ret[1]
    translation_vector = ret[2]

    f_2d = np.linalg.inv(K2) @ np.concatenate((f2d[:, 0],
                                               np.ones((f2d.shape[0], 1))), axis=1).T

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    translation_vector = translation_vector.reshape(3)
    proj = rotation_mat @ f3d[:, 0].T + translation_vector.reshape(3, -1)
    proj = proj[:2] / proj[2:]
    reproj_error = np.linalg.norm(f_2d[:2] - proj[:2], axis=0)
    reproj_inliers = reproj_error < 5e-2
    reproj_inliers = reproj_inliers.reshape(-1)

    if success==0 or reproj_inliers.sum() < 10:
        return 0, None, None, None
    else:
        ret = cv2.solvePnP(f3d[reproj_inliers].reshape(reproj_inliers.sum(), 1, 3),
                           f2d[reproj_inliers].reshape(reproj_inliers.sum(), 1, 2),
                           K2,
                           distCoeffs=None,
                           flags=cv2.SOLVEPNP_ITERATIVE)
        success = ret[0]
        rotation_vector = ret[1]
        translation_vector = ret[2]

        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        translation_vector = translation_vector.reshape(3)

        Caz_T_Wmp = np.eye(4)
        Caz_T_Wmp[:3, :3] = rotation_mat
        Caz_T_Wmp[:3, 3] = translation_vector

        rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        translation_vector = translation_vector.reshape(3)
        proj = rotation_mat @ f3d[:, 0].T + translation_vector.reshape(3, -1)
        proj = proj[:2] / proj[2:]
        reproj_error_refined = np.linalg.norm(f_2d[:2] - proj[:2], axis=0)
        reproj_error_refined = reproj_error_refined < 1e-2
        reproj_error_refined = reproj_error_refined.reshape(-1)

        if reproj_error_refined.sum() < 0.8 * reproj_inliers.sum():
            return 0, None, None, None
        else:
            return success, Caz_T_Wmp, f2d[reproj_error_refined, 0], f3d[reproj_error_refined, 0]


def valid_2Dfeatures(x):
    return (x[:, 0] != -1) * (x[:, 1] != -1)


def valid_3Dfeatures(X):
    return (X[:, 0] != -1) * (X[:, 1] != -1) * (X[:, 2] != -1)


def SetupPnPNL(P, X, track):
    n_points = X.shape[0]
    n_projs = np.sum(track[:, 0] != -1)
    b = np.zeros((2 * n_projs,))

    S_row = []
    S_col = []

    k = 0
    point_index = []

    for j in range(n_points):
        if track[j, 0] != -1 and track[j, 1] != -1:
            # S[2*k : 2*(k+1), 7*i : 7*(i+1)] = 1
            rows, cols = np.meshgrid(np.linspace(2 * k, 2 * (k + 1), num=2, endpoint=False, dtype=int),
                                     np.linspace(0, 7, num=7, endpoint=False, dtype=int))
            rows = rows.reshape(-1)
            cols = cols.reshape(-1)
            S_row.append(rows)
            S_col.append(cols)

            b[2 * k: 2 * (k + 1)] = track[j, :]
            point_index.append(j)
            k += 1


    point_index = np.asarray(point_index)

    S_row = np.concatenate(S_row)
    S_col = np.concatenate(S_col)
    S_data = np.ones(S_row.shape[0], dtype=bool)
    S = csr_matrix((S_data, (S_row, S_col)), shape=(2 * n_projs, 7 + 3 * n_points))

    z = np.zeros((7 + 3 * n_points,))
    R = P[:, :3]
    t = P[:, 3]
    q = Rotation2Quaternion(R)
    p = np.concatenate([t, q])
    z[0:7] = p
    for i in range(n_points):
        z[7 + 3 * i: 7 + 3 * (i + 1)] = X[i, :]

    return z, b, S, point_index


def MeasureReprojectionSinglePose(z, b, point_index):

    n_projs = point_index.shape[0]
    f = np.zeros((2 * n_projs,))
    p = z[0:7]
    q = p[3:]
    q_norm = np.sqrt(np.sum(q ** 2))
    q = q / q_norm
    R = Quaternion2Rotation(q)
    t = p[:3]

    for k, j in enumerate(point_index):
        X = z[7 + 3 * j: 7 + 3 * (j + 1)]
        # Remove measurement error of fixed poses
        proj = R @ X + t
        proj = proj / proj[2]
        f[2 * k: 2 * (k + 1)] = proj[:2]

    err = b - f

    return err


def UpdatePose(z):

    p = z[0:7]
    q = p[3:]

    q = q / np.linalg.norm(q)
    R = Quaternion2Rotation(q)
    t = p[:3]
    P_new = np.hstack([R, t[:, np.newaxis]])

    return P_new


def RunPnPNL(P, X, track):

    z0, b, S, point_index = SetupPnPNL(P, X, track)
    # print('starting optimization')
    # print('feature: ', track.shape[0])
    # print('nnz: ', S.nnz)
    res = least_squares(
        lambda x: MeasureReprojectionSinglePose(x, b, point_index),
        z0,
        jac_sparsity=S,
        verbose=0,
        ftol=1e-4,
        max_nfev=50,
        xtol=1e-5,
        loss='huber',
        f_scale=0.01
    )
    # loss = 'soft_l1',
    # f_scale = 0.1
    z = res.x

    # err0 = MeasureReprojectionSinglePose(z0, b, point_index)
    # err = MeasureReprojectionSinglePose(z, b, point_index)
    # print('Reprojection error {} -> {}'.format(np.linalg.norm(err0), np.linalg.norm(err)))

    P_new = UpdatePose(z)

    return P_new



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Incremental SFM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--ego_dataset_folder', type=str, default='',
        help='Ego dataset folder')
    parser.add_argument(
        '--viz', action='store_true',
        help='Reprojection visualization')

    opt = parser.parse_args()

    EGOLOC_FOLDER = opt.ego_dataset_folder
    INPUT_DIR = os.path.join(EGOLOC_FOLDER, 'track')

    track = np.load(os.path.join(INPUT_DIR, 'track.npy'))[:, ::2]
    original_image_id_list = np.load(os.path.join(INPUT_DIR, 'original_image_id.npy'))
    print('original image id list: ', len(original_image_id_list))
    print('track: ', track.shape)

    # depth = np.load(os.path.join(INPUT_DIR, 'depth_adaptive_20_2.npy'))
    num_images = track.shape[0]

    # Load input images
    OUTPUT_DIR = os.path.join(INPUT_DIR, 'poses')

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    ####
    P = np.load(os.path.join(EGOLOC_FOLDER, 'poses_reloc', 'camera_poses_pnp.npy'))
    good_pose_pnp = np.load(os.path.join(EGOLOC_FOLDER, 'poses_reloc', 'good_pose_pnp.npy'))
    fixed_pose_parameters = copy.deepcopy(good_pose_pnp)

    # Triangulate initial structure
    valid_idx = np.sum((track[:, :, 0] != -1) *
                       (track[:, :, 1] != -1), axis=0) >= 3
    track = track[:, valid_idx]
    print('tracks with length > 3: ', track.shape[1])
    X = -np.ones((track.shape[1], 3))

    # Enriching the structure by iterative triangulation and PnP
    for trial in range(5):
        valid_pose_triangulation_ransac = copy.deepcopy(good_pose_pnp)
        for f_idx in range(track.shape[1]):
            ## When depth map is not available
            cam_idx = (track[:, f_idx, 0] != -1) * (track[:, f_idx, 1] != -1) * good_pose_pnp
            num_observed_poses = np.sum(cam_idx)
            if num_observed_poses >= 3:
                uv1 = np.concatenate((track[cam_idx, f_idx], np.ones((num_observed_poses, 1))), axis=1)
                _, inlier, outlier = Triangulation_RANSAC(uv1, P[cam_idx], threshold=5e-2)
                if len(inlier) >= 3:
                    C1_X = Triangulation_LS(uv1[inlier], P[cam_idx][inlier])
                    X[f_idx] = C1_X
                    valid_pose_triangulation_ransac[cam_idx][outlier] = False

        print('Triangulated features ', np.sum(valid_3Dfeatures(X)))
        print('good poses/total poses ', np.sum(valid_pose_triangulation_ransac), '/', good_pose_pnp.shape[0])
        good_pose_pnp = copy.deepcopy(valid_pose_triangulation_ransac)

        ## Refine/Re-estimate bad poses with PnP
        for cam_idx in range(num_images):
            if good_pose_pnp[cam_idx] == 0:
                feature_idx_in_cam_idx = valid_2Dfeatures(track[cam_idx]) * valid_3Dfeatures(X)
                # print('cam idx ', cam_idx, ' visible features: ', np.sum(feature_idx_in_cam_idx))

                if np.sum(feature_idx_in_cam_idx) < 20:
                    # print('Poses ', cam_idx, ' has only ', np.sum(feature_idx_in_cam_idx), ' matches to surrounding structure')
                    continue

                # print('Poses ', cam_idx, ' has ', np.sum(feature_idx_in_cam_idx),
                #       ' matches to surrounding structure')
                pts3d = np.expand_dims(X[feature_idx_in_cam_idx], axis=1)
                pts2d = np.expand_dims(track[cam_idx, feature_idx_in_cam_idx], axis=1)

                # pnp ransac
                ret = cv2.solvePnPRansac(pts3d,
                                         pts2d,
                                         np.eye(3),
                                         distCoeffs=None,
                                         flags=cv2.SOLVEPNP_EPNP)

                success = ret[0]
                rotation_vector = ret[1]
                translation_vector = ret[2]
                inliers_ids = ret[3]
                rotation_mat, _ = cv2.Rodrigues(rotation_vector)
                translation_vector = translation_vector.reshape(3)
                P_init = np.zeros((3, 4))
                P_init[:, :3] = rotation_mat
                P_init[:, 3] = translation_vector

                if success and len(inliers_ids) > 10:
                    P_new = RunPnPNL(P_init, X[feature_idx_in_cam_idx],
                                     track[cam_idx, feature_idx_in_cam_idx])

                    f_2d = np.concatenate((track[cam_idx, feature_idx_in_cam_idx],
                                           np.ones((feature_idx_in_cam_idx.sum(), 1))), axis=1).T

                    proj = P_new[:, :3] @ X[feature_idx_in_cam_idx].T + P_new[:, 3:]
                    proj = proj[:2] / proj[2:]

                    reproj_error = np.linalg.norm(f_2d[:2] - proj[:2], axis=0)
                    reproj_inliers = reproj_error < 5e-2

                    if np.sum(reproj_inliers) > 10:
                        P[cam_idx] = P_new
                        good_pose_pnp[cam_idx] = 1

        print('good poses/total poses ', np.sum(good_pose_pnp), '/', good_pose_pnp.shape[0])
        WritePosesToPly(P[good_pose_pnp], '%s/cameras_pnp_triangulate_iter_%d.ply' % (OUTPUT_DIR, trial))


    ## Return all poses for BA
    np.save('%s/cameras_pnp_triangulation.npy' % OUTPUT_DIR, P)
    np.save('%s/features_pnp_triangulation.npy' % OUTPUT_DIR, X)
    np.save('%s/track_pnp_triangulation.npy' % OUTPUT_DIR, track)
    np.save('%s/original_image.npy' % OUTPUT_DIR, original_image_id_list)
    np.save('%s/fixed_poses_ids_pnp_triangulation.npy' % OUTPUT_DIR, fixed_pose_parameters)
    np.save('%s/good_poses_ids_pnp_triangulation.npy' % OUTPUT_DIR, good_pose_pnp)
    WritePosesToPly(P[good_pose_pnp], '%s/cameras_pnp_triangulation.ply' % OUTPUT_DIR)


    #### Check reprojection error
    P = np.load('%s/cameras_pnp_triangulation.npy' % OUTPUT_DIR)
    X = np.load('%s/features_pnp_triangulation.npy' % OUTPUT_DIR)
    track = np.load('%s/track_pnp_triangulation.npy' % OUTPUT_DIR)
    original_image_id_list = np.load('%s/original_image.npy' % OUTPUT_DIR)
    good_pose_pnp = np.load('%s/good_poses_ids_pnp_triangulation.npy' % OUTPUT_DIR)

    # fixed_pose_parameters = np.load('%s/fixed_poses_ids_pnp_triangulation.npy' % OUTPUT_DIR)
    # good_pose_pnp = np.load('%s/good_poses_ids_pnp_triangulation.npy' % OUTPUT_DIR)
    print('good pose reproj ', good_pose_pnp.sum())
    good_pose_reprojection = copy.deepcopy(good_pose_pnp)
    for i in range(num_images):
        if good_pose_reprojection[i]:
            feature_idx_in_cam_idx = valid_2Dfeatures(track[i]) * valid_3Dfeatures(X)
            reliable_tracks = np.sum((track[:, feature_idx_in_cam_idx, 0] != -1) *
                                     (track[:, feature_idx_in_cam_idx, 1] != -1), axis=0) >= 3

            if reliable_tracks.sum() < 20:
                good_pose_reprojection[i] = False
            else:
                f_2d = np.concatenate((track[i, feature_idx_in_cam_idx][reliable_tracks],
                                        np.ones((reliable_tracks.sum(), 1))), axis=1).T

                proj = P[i, :, :3] @ X[feature_idx_in_cam_idx][reliable_tracks].T + P[i, :, 3:]
                camera_frontal = proj[2] > 0
                proj = proj[:2] / proj[2:]

                reproj_error = np.linalg.norm(f_2d[:2] - proj[:2], axis=0)
                reproj_inliers = (reproj_error < 5e-2) * (camera_frontal)

                if reproj_inliers.sum() < 20:
                    good_pose_reprojection[i] = False

            # print('cam_idx ', i, ' good pose? ', good_pose_reprojection[i])
            if opt.viz:
                K_ego = np.loadtxt('%s/intrinsics.txt' % opt.ego_dataset_folder)
                uv1 = np.concatenate((track[i, feature_idx_in_cam_idx][reliable_tracks],
                                      np.ones((reliable_tracks.sum(), 1))), axis=1)
                cam_img = cv2.imread(os.path.join(EGOLOC_FOLDER,
                                                  'color/color_%07d.jpg') % original_image_id_list[i])
                VisualizeReprojectionError(P[i],
                                           X[feature_idx_in_cam_idx][reliable_tracks],
                                           uv1,
                                           Im=cam_img, K=K_ego)

    np.save('%s/good_pose_reprojection.npy' % OUTPUT_DIR, good_pose_reprojection)
    WritePosesToPly(P[good_pose_reprojection], '%s/cameras_poses_final.ply' % OUTPUT_DIR)
    print('good pose reproj ', good_pose_reprojection.sum())