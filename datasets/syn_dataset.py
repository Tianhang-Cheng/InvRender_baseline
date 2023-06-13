import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util 
from glob import glob
import cv2 as cv
from pathlib import Path
import imageio.v3 as imageio
import pdb
inv_tone_map = lambda x: np.power(x, 2.2)

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class SynDataset(torch.utils.data.Dataset):
    def __init__(self,
                 instance_dir,
                 frame_skip=0,
                 split='train'
                 ):  
        self.instance_dir = instance_dir 

        self.split = split
        self.device = 'cuda'   

        if split == 'test':
            # our dataset
            instance_dir = instance_dir.replace('_neus', '_neus_test')
            # synthetic4relight
            # instance_dir = instance_dir.replace('synth4relight_neus', 's4r_test')
        
        # our dataset
        self.data_dir = instance_dir
        if 'antman' in instance_dir:
            self.mask_dir = ' dataset/antman/test/inputs'
        elif 'apple' in instance_dir:
            self.mask_dir = ' dataset/apple/test/inputs'
        elif 'chest' in instance_dir:
            self.mask_dir = ' dataset/chest/test/inputs'
        elif 'tpiece' in instance_dir:
            self.mask_dir = ' dataset/tpiece/test/inputs'
        elif 'gamepad' in instance_dir:
            self.mask_dir = ' dataset/gamepad/test/inputs'
        elif 'ping_pong_racket' in instance_dir:
            self.mask_dir = ' dataset/ping_pong_racket/test/inputs'
        elif 'porcelain_mug' in instance_dir:
            self.mask_dir = ' dataset/porcelain_mug/test/inputs'
        elif 'wood_bowl' in instance_dir:
            self.mask_dir = ' dataset/wood_bowl/test/inputs'
        
        # synthetic4relight
        elif 'hotdog' in instance_dir:
            self.mask_dir = ' synth4relight_subsampled/hotdog/inputs'
        elif 'chair' in instance_dir:
            self.mask_dir = ' synth4relight_subsampled/chair/inputs'
        elif 'jugs' in instance_dir:
            self.mask_dir = ' synth4relight_subsampled/jugs/inputs'
        elif 'air_baloons' in instance_dir:
            self.mask_dir = ' synth4relight_subsampled/air_baloons/inputs'
        else:
            raise ValueError
        
        if split == 'test':
            self.mask_dir = self.mask_dir.replace('/inputs','')

        self.render_cameras_name = 'cameras_sphere.npz'
        self.object_cameras_name = 'cameras_sphere.npz'

        self.camera_outside_sphere = True
        self.scale_mat_scale = 1.1

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis) 
        self.images_np = np.stack([imageio.imread(im_name) for im_name in self.images_lis]) / 255.0
        self.images_np = inv_tone_map(self.images_np)
        if split == 'test':
            self.masks_lis = sorted(glob(os.path.join(self.mask_dir, 'gt_mask_*.png'))) 
            self.masks_np = np.ones_like(self.images_np)[..., 0].astype(float)
        else:
            self.masks_lis = sorted(glob(os.path.join(self.mask_dir, 'mask_binary_*.png'))) 
            self.masks_np = np.stack([imageio.imread(im_name) for im_name in self.masks_lis]) / 255.0
            self.masks_np = self.masks_np[..., 0]  

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W ]
        self.masks = (self.masks > 0.5).bool() 
        self.intrinsics_all = torch.stack(self.intrinsics_all)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all)   # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('laod {} images'.format(self.n_images))
        print('Load data: End')  
        self.n_cameras = self.n_images  
             
        self.img_res = (self.H, self.W)
        self.total_pixels = self.H * self.W

        # pdb.set_trace()

        # self.rgb_images = []
        # self.object_masks = []
        # image_paths = glob(os.path.join(self.instance_dir,  'image_*.png'))[::frame_skip]
        # image_paths.sort()
        # mask_paths = glob(os.path.join(self.instance_dir,  'mask_*.png'))[::frame_skip]
        # mask_paths.sort() 

        # read training images
        self.rgb_images = self.images
        self.object_masks = self.masks   

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None 
 
 
    def __len__(self):
        return (self.n_images)

    def __getitem__(self, idx):
        if self.single_imgname_idx is not None:
            idx = self.single_imgname_idx
        
        uv = np.mgrid[0:self.H, 0:self.W].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)   
        px = uv[:, 0]
        py = uv[:, 1]
  
        p = torch.stack([px, py, torch.ones_like(py)], dim=-1).float().cpu()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3 
 
        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx], 
            "object_mask": self.masks[idx].reshape(-1,1), 
            "rays_o": rays_o ,
            "rays_d": rays_v ,
        }  
        ground_truth = {
            "rgb": self.images[idx].reshape(-1,3), 
        }  
 
        # if self.split == 'test':
        #     ground_truth["envmap6_rgb"] = self.envmap6_images[idx]
        #     ground_truth["envmap12_rgb"] = self.envmap12_images[idx]
 
        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.images[idx].reshape(-1,3)[self.sampling_idx, :]
            sample["object_mask"] = self.masks[idx].reshape(-1,1)[self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]
            sample["rays_o"] = rays_o[self.sampling_idx, :]
            sample["rays_d"] = rays_v[self.sampling_idx, :]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, 
        # ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]


