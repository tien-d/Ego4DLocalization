
import argparse
import sys
import os
from PIL import Image
import numpy as np
import open3d as o3d
import glob
from matplotlib import pyplot as plt
import cv2
import json
import fnmatch


def create_traj_azure(output_traj, Ci_T_G=None):

    K_mp = [457.1812438964844, 0., 0., 0., 478.614013671875, 0., 480. - 0.5, 270. - 0.5, 1.]

    d = json.load(open('/home/tien/Code/Ego4DLocalization/Localization/camera_trajectory.json', 'r'))
    dp0 = d['parameters'][0]
    dp0['intrinsic']['width'] = 960
    dp0['intrinsic']['height'] = 540
    dp0['intrinsic']['intrinsic_matrix'] = K_mp
    dp0['extrinsic'] = []
    x = []


    if Ci_T_G is not None:
        for i in range(Ci_T_G.shape[0]):
            temp = dp0.copy()

            # E = np.linalg.inv(G_T_Ci[i])
            E = Ci_T_G[i]

            E_v = np.concatenate([E[:, i] for i in range(4)], axis=0)
            temp['extrinsic'] = E_v.tolist()
            x.append(temp)

    d['parameters'] = x
    with open(output_traj, 'w') as f:
        json.dump(d, f)


def custom_draw_geometry_with_camera_trajectory(pcd, output_path='', input_image_folder='',
                                                comparison_color='', trajectory='',
                                                G_T_Ci=None, original_img_indices=None):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory = \
        o3d.io.read_pinhole_camera_trajectory(trajectory)
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()


    if not os.path.exists(os.path.join(output_path)):
        os.makedirs(os.path.join(output_path))

    def setup(vis):
        ctr = vis.get_view_control()
        ctr.set_zoom(0.450)
        ctr.rotate(0.0, -4e2)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory

        if glb.index >= 0 and glb.index < len(original_img_indices):
            print("Capture image {:05d}".format(glb.index))
            image = vis.capture_screen_float_buffer(False)

            image = np.asarray(image)
            if output_path != '':
                color_info = os.path.join(input_image_folder, 'color/color_%07d.jpg' % original_img_indices[glb.index])
                color_img = Image.open(color_info)
                color_img = color_img.resize((960, 540), resample=Image.BILINEAR)

                img = (255. * image)
                added_image = cv2.addWeighted(np.asarray(color_img), 0.8, img.astype(np.uint8), 0.5, 0)
                full_viz_image = np.concatenate([np.asarray(color_img), img.astype(np.uint8), added_image], axis=1)
                cv2.imwrite(os.path.join(output_path, 'render_%07d.png' % glb.index),
                            cv2.cvtColor(full_viz_image, cv2.COLOR_RGB2BGR))


        glb.index += 1

        if glb.index < len(original_img_indices): # visualize only well estimated views
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            exit(1)

        return False


    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=960, height=540)
    vis.add_geometry(pcd)
    setup(vis)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    ROOT_FOLDER = './data/egovideo'
    INPUT_DIR = os.path.join(ROOT_FOLDER, 'poses_reloc')

    original_image_ids = np.arange(0, len(
        fnmatch.filter(os.listdir(os.path.join(ROOT_FOLDER, 'color')), '*.jpg')), step=1)
    valid_pose = np.load(os.path.join(INPUT_DIR, 'good_pose_pnp.npy'))
    C_T_G = np.load(os.path.join(INPUT_DIR, 'camera_poses_pnp.npy'))
    print('C_T_G ', C_T_G.shape)

    Ci_T_G = np.zeros((len(original_image_ids), 4, 4))
    k = 0
    for i in range(len(original_image_ids)):
        if valid_pose[i]:
            Ci_T_G[k] = np.concatenate((C_T_G[i], np.array([[0., 0., 0., 1.]])), axis=0)
            k += 1

    print("Create trajectory ...")
    create_traj_azure(output_traj='./egovideo_camera_traj.json',
                      Ci_T_G=Ci_T_G)

    print("Loading point cloud ...")
    SAVING_SUBFOLDER = 'pose_visualization'
    if not os.path.exists(os.path.join(ROOT_FOLDER, SAVING_SUBFOLDER)):
        os.makedirs(os.path.join(ROOT_FOLDER, SAVING_SUBFOLDER))

    mesh_file = fnmatch.filter(os.listdir(os.path.join('./data', 'scan', 'matterpak')), '*.obj')[0]
    mesh = o3d.io.read_triangle_mesh(os.path.join('./data', 'scan', 'matterpak', mesh_file), enable_post_processing=True)
    custom_draw_geometry_with_camera_trajectory(mesh, output_path=os.path.join(ROOT_FOLDER, SAVING_SUBFOLDER),
                                                input_image_folder=ROOT_FOLDER,
                                                trajectory='./egovideo_camera_traj.json',
                                                G_T_Ci=Ci_T_G,
                                                original_img_indices=original_image_ids[valid_pose])