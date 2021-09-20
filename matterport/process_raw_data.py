import numpy as np
import os
import fnmatch
import base64
from io import BytesIO
from PIL import Image

K_mp = [700., 0., 0., 0., 700., 0., 640. - 0.5, 480. - 0.5, 1.]
DATA_FOLDER = ''

if not os.path.exists(os.path.join(DATA_FOLDER, 'color')):
    os.makedirs(os.path.join(DATA_FOLDER, 'color'))

if not os.path.exists(os.path.join(DATA_FOLDER, 'pose')):
    os.makedirs(os.path.join(DATA_FOLDER, 'pose'))

pose_files = fnmatch.filter(os.listdir(os.path.join(DATA_FOLDER, 'raw')), '*_pose.txt')

for i in range(len(pose_files)):
    E = np.identity(4, dtype=float)

    pose = np.loadtxt(DATA_FOLDER + '/raw/%d_pose.txt' % i)
    print(' i = ', i, ' | pose = ', pose)

    E[:, 0] = np.array([1, 0, 0, 0])
    E[:, 1] = np.array([0, 0, -1, 0])
    E[:, 2] = np.array([0, 1, 0, 0])
    E[:, 3] = np.array([pose[0], -pose[2], pose[1], 1])

    pose[3:] *= np.pi/180

    roty = np.array([[np.cos(pose[4]), 0., -np.sin(pose[4]), 0.],
                     [0.,              1.,       0.,         0.],
                     [np.sin(pose[4]), 0., np.cos(pose[4]),  0.],
                     [0.,              0.,       0.,         1.]])

    rotx = np.array([[1., 0., 0., 0.],
                     [0., np.cos(pose[3]), -np.sin(pose[3]), 0.],
                     [0., np.sin(pose[3]), np.cos(pose[3]), 0.],
                     [0.,  0., 0.,  1.]])

    G_T_C = E @ roty @ rotx
    np.savetxt(DATA_FOLDER + '/pose/pose_%06d.txt' % i, G_T_C)


    with open(DATA_FOLDER + '/raw/%d_rgb.txt' % i, 'r') as file:
        data = file.read()

    im = Image.open(BytesIO(base64.b64decode(data[22:])))
    im.save(DATA_FOLDER + '/color/color_%06d.jpg' % i, 'JPEG')
