import argparse
import sys
import os
from PIL import Image
import numpy as np
import open3d as o3d
import glob
from matplotlib import pyplot as plt
import fnmatch

ROOT_FOLDER = ''
DATA_FOLDER = ''
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
FOCAL_LENGTH = 700.

import json
import numpy as np
from pyquaternion import Quaternion
import os


def create_trajectory(pose_folder, output_path):
    K_mp = [FOCAL_LENGTH, 0., 0., 0., FOCAL_LENGTH, 0., 0.5*SCREEN_WIDTH - 0.5, 0.5*SCREEN_HEIGHT - 0.5, 1.]

    d = json.load(open('./camera_trajectory.json', 'r'))
    dp0 = d['parameters'][0]
    dp0['intrinsic']['width'] = SCREEN_WIDTH
    dp0['intrinsic']['height'] = SCREEN_HEIGHT
    dp0['intrinsic']['intrinsic_matrix'] = K_mp
    dp0['extrinsic'] = []
    x = []

    file_list = sorted(os.listdir(pose_folder))
    for i in range(len(file_list)):
        temp = dp0.copy()
        print(os.path.join(pose_folder, file_list[i]))
        G_T_C = np.loadtxt(os.path.join(pose_folder, file_list[i]))
        E = np.linalg.inv(G_T_C)
        E_v = np.concatenate([E[:, i] for i in range(4)], axis=0)
        temp['extrinsic'] = E_v.tolist()
        x.append(temp)


    d['parameters'] = x
    with open(output_path, 'w') as f:
        json.dump(d, f)

    return len(file_list)


def custom_draw_geometry_with_camera_trajectory(pcd, trajectory='', N_iter=0, output_folder='./'):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory = \
        o3d.io.read_pinhole_camera_trajectory(trajectory)
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()



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


        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            image = vis.capture_screen_float_buffer(False)

            color_info = '/home/kvuong/dgx-projects/ego4d_data/Matterport/%s/color/color_%06d.jpg' % (DATA_FOLDER, glb.index)
            color_img = Image.open(color_info)
            color_img = color_img.resize((SCREEN_WIDTH, SCREEN_HEIGHT), resample=Image.BILINEAR)
            color_img.save(output_folder.replace('depth', 'scolor') + '/color_%06d.png' % glb.index)

            depth_image = vis.capture_depth_float_buffer(False)
            depth_image = Image.fromarray((np.array(depth_image) * 1000).astype(np.uint32))
            depth_image.save(output_folder + '/depth_%06d.png' % glb.index)
            # plt.imsave('/home/tiendo/Data/Matterport/bathroom_605/depths/depth_%06d.png' % glb.index, depth_image, dpi=1)
            # plt.imsave(os.path.join(save_color, "{:05d}.png".format(glb.index)), image, dpi=1)

        glb.index += 1

        if glb.index < N_iter: # visualize only 500 views
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index], allow_arbitrary=True)
        else:
            vis.register_animation_callback(None)

        return False


    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, visible=False) # width=SCREEN_WIDTH, height=SCREEN_HEIGHT
    vis.add_geometry(pcd)
    setup(vis)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':

    input_mesh = fnmatch.filter(os.listdir(os.path.join(ROOT_FOLDER, DATA_FOLDER, 'matterpak')), '*.obj')
    input_mesh = os.path.join(ROOT_FOLDER, DATA_FOLDER, 'matterpak', input_mesh[0])
    print(input_mesh)

    output_traj = os.path.join(ROOT_FOLDER, DATA_FOLDER, 'scans_poses.json')
    N_iter = create_trajectory(os.path.join(ROOT_FOLDER, DATA_FOLDER, 'pose'), output_traj)


    output_folder = os.path.join(ROOT_FOLDER, DATA_FOLDER, 'depth')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mesh = o3d.io.read_triangle_mesh(input_mesh)
    custom_draw_geometry_with_camera_trajectory(mesh, trajectory=output_traj, N_iter=N_iter,
                                                output_folder=output_folder)
