import torch
import cv2

from models.superpoint import SuperPoint
import argparse
import random
import numpy as np
import matplotlib.cm as cm

from torch.utils.data.dataset import Dataset
import pickle
import numpy as np
from PIL import Image
import os
from torch.utils.data import DataLoader

from models.utils import read_image


class MatterportDataset(Dataset):
    def __init__(self, dataset_name='walterlib', resize=[640, 480]):
        super(MatterportDataset, self).__init__()
        self.root_dir = '/home/tiendo/Data/Matterport'
        self.data_path = os.path.join(self.root_dir, dataset_name, 'scolor')
        self.data_info = os.listdir(self.data_path)
        self.data_len = len(self.data_info)
        self.resize = resize

    def __getitem__(self, index):
        color_info = os.path.join(self.data_path, self.data_info[index])
        print(color_info)
        _, gray_tensor, _ = read_image(color_info, 'cpu', resize=self.resize, rotation=0, resize_float=False)
        gray_tensor = gray_tensor.reshape(1, 480, 640)

        color_img = Image.open(color_info)
        color_tensor = torch.tensor(np.asarray(color_img))[None, ...]
        output = {'image': gray_tensor, 'cimage': color_tensor}

        return output

    def __len__(self):
        return self.data_len



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

superpoint = SuperPoint(config.get('superpoint', {}))

matterport_dataset = MatterportDataset()
data_loader = DataLoader(dataset=matterport_dataset,
                            num_workers=1, batch_size=4, shuffle=False,
                            pin_memory=True)

for idx, images in enumerate(data_loader):
    output = superpoint(images)
    print(output['descriptors'][0].shape)
    print(output['scores'][0].shape)

    kpts = [cv2.KeyPoint(output['keypoints'][0][k, 0] * 3.0, output['keypoints'][0][k, 1] * 2.25, 50) for k in range(output['keypoints'][0].shape[0])]


    input_img = images['cimage'][0, 0].detach().clone().cpu().numpy()
    input_img = input_img.astype(np.uint8)
    kp_img = images['cimage'][0, 0].detach().clone().cpu().numpy()
    kp_img = kp_img.astype(np.uint8)
    print(kp_img.shape)
    cv2.drawKeypoints(input_img, kpts, kp_img, color=(0, 255, 0))
    print(kp_img.shape)
    cv2.imwrite('kp_viz/%d.png' % idx, cv2.cvtColor(kp_img, cv2.COLOR_RGB2BGR))

