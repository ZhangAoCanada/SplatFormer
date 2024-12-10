import torch
import gin
import numpy as np 
from typing import Optional, Union
import pickle
import cv2, os
from utils.transform_utils import remove_outliers, MinMaxScaler
from dataset import colmap_utils
from utils import gs_utils
import glob, yaml
from pathlib import Path
import random
from PIL import Image
from time import time

from scene import Scene
import torch.nn as nn
import torch.nn.functional as F
from scene.utils.general_utils import inverse_sigmoid, PILtoTorch
from scene.utils.sh_utils import RGB2SH
from scene.utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal, getWorld2View, getWorld2View2, getProjectionMatrix
from simple_knn._C import distCUDA2
from PIL import Image
import random
from plyfile import PlyData, PlyElement


@gin.configurable
class SplatfactoDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 train_or_test,
                 nerfstudio_folder,
                 colmap_folder,
                 load_pose_src, #[colmap or nerfstudio]
                 sample_ratio_test: Optional[float],
                 image_per_scene: Optional[int],
                 remove_outlier_ndevs: float,
                 max_gs_num: int,
                 cache_steps: int,
                 cache_num_scenes: int, #Default: cache_num_scenes=1, cache_steps=1
                 split_across_gpus: bool,
                 background_color: list=[0,0,0],
                 ):
        self.train_or_test = train_or_test
        self.remove_outlier_ndevs = remove_outlier_ndevs
        self.background_color = background_color
        # self.image_per_scene = image_per_scene
        # self.sample_ratio_test = sample_ratio_test
        # self.max_gs_num = max_gs_num
        self.image_per_scene = 9
        self.sample_ratio_test = 0.5
        self.max_gs_num = 60000

        self.sh_degree = 1
        self.model_path="outputs/truck/trying_debug"
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.source_path="data/tandt_db/tandt/truck"
        self.scene = Scene(model_path=self.model_path, source_path=self.source_path)
        self.train_caminfos = self.scene.getTrainCameraInfos()
        self.test_caminfos = self.scene.getTestCameraInfos()
        self.scene = self.get_scene_from_cache()

 
    @gin.configurable
    def read_image(self, path, background):
        try:
            pil_image = Image.open(path)
        except:
            print(f'Warning: {path} cannot be opened')

        image = np.array(pil_image, dtype="uint8").astype(np.float32) / 255.0
        if 'real' in path.lower():
            possible_mask_filename = path.replace('images','masks') #TODO: hardcoded
            if os.path.exists(possible_mask_filename):
                mask = np.array(Image.open(possible_mask_filename)).astype(image.dtype)/255.0
                mask = torch.from_numpy(mask)
        else:
            mask = None

        image = torch.from_numpy(image)
        if image.shape[2] == 4:
            image = image[:, :, :3] * image[:, :, -1:] + background * (1.0 - image[:, :, -1:])
        elif mask is not None:
            image_rgb = image * mask[...,None] + background * (1.0 - mask[...,None])
            # As we need to preserve the mask for evaluation, we save the image RGBA
            image = torch.concat([image_rgb, mask[...,None]], axis=-1) # Hardcoded here, only for the real dataset
        return image


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        gs_params = {}
        gs_params["means"] = torch.tensor(xyz, dtype=torch.float)
        gs_params["features_dc"] = torch.tensor(features_dc, dtype=torch.float).transpose(1, 2).contiguous().squeeze(1)
        gs_params["features_rest"] = torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous()
        gs_params["opacities"] = torch.tensor(opacities, dtype=torch.float)
        gs_params["scales"] = torch.tensor(scales, dtype=torch.float)
        gs_params["quats"] = torch.tensor(rots, dtype=torch.float)

        # Remove inf or nan
        select = torch.ones(gs_params['means'].shape[0], dtype=torch.bool)
        for key in gs_params:
            if key=='features_rest':
                select = select & ~torch.isnan(gs_params[key].sum(dim=1)).any(dim=1)
            else:
                select = select & ~torch.isnan(gs_params[key]).any(dim=1)
        for key in gs_params:
            gs_params[key] = gs_params[key][select]

        # Filter the outliers
        if self.remove_outlier_ndevs > 0:
            _,inlier_mask = remove_outliers(gs_params['means'], n_devs=self.remove_outlier_ndevs)
            for key in gs_params:
                gs_params[key] = gs_params[key][inlier_mask]

        # # Truncate gs params if num > self.max_gs_num
        # N = gs_params['means'].shape[0]
        # if N > self.max_gs_num:
        #     inlier_mask = torch.zeros(N, dtype=torch.bool)
        #     inlier_mask[:self.max_gs_num] = True
        #     for key in gs_params:
        #         gs_params[key] = gs_params[key][inlier_mask]

        # # Normalize the means and scales (we need to use the scaler to transform the camera later)
        # scaler = MinMaxScaler()
        # gs_params['means'] = scaler.fit_transform(gs_params['means']) 
        # gs_params['scales'] = gs_params['scales'] + torch.log(scaler.scale_)

        # inf_mask = torch.isinf(gs_params['scales']).sum(dim=1).bool()
        # valid_mask = (~inf_mask).bool()
        # inrange_mask = torch.all((gs_params['means'] >= 0) & (gs_params['means'] <= 1), dim=1)
        # valid_mask = valid_mask & inrange_mask
        # for key in gs_params:
        #     gs_params[key] = gs_params[key][valid_mask]
        #     if torch.isnan(gs_params[key]).any():
        #         print(f'Warning: {key} contains nan')

        scaler = None

        return gs_params, scaler


    def load_gs_params(self):
        pcd = self.scene.point_cloud
        spatial_lr_scale = self.scene.cameras_extent
        gs_params = {}
        self.spatial_lr_scale = spatial_lr_scale

        ################# NOTE: create from pcd ##########################
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float())
        features = torch.zeros((fused_color.shape[0], 3, (self.sh_degree + 1) ** 2)).float()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()).cpu(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))
        ################# NOTE: create from random #######################
        # num_init_pnts = 40000
        # dist_range = 200.
        # fused_point_cloud = ((torch.rand((num_init_pnts, 3), dtype=torch.float, ) - 0.5) * 2.0) * dist_range
        # fused_color = RGB2SH(torch.rand((num_init_pnts, 3), dtype=torch.float))
        # features = torch.zeros((fused_color.shape[0], 3, (self.sh_degree + 1) ** 2), dtype=torch.float)
        # features[:, :3, 0 ] = fused_color
        # features[:, 3:, 1:] = 0.0
        # print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # scales = torch.log(torch.rand((fused_point_cloud.shape[0], 3), dtype=torch.float) * 1 + 0.0001)
        # rots = torch.zeros((fused_point_cloud.shape[0], 4), dtype=torch.float)
        # rots[:, 0] = 1
        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))
        ##################################################################

        gs_params['means'] = torch.Tensor(fused_point_cloud)
        # gs_params['features_dc'] = torch.Tensor(features[:,:,0:1].transpose(1, 2).contiguous())
        gs_params['features_dc'] = torch.Tensor(features[:,:, 0].contiguous())
        gs_params['features_rest'] = torch.Tensor(features[:,:,1:].transpose(1, 2).contiguous())
        gs_params['scales'] = torch.Tensor(scales)
        gs_params['quats'] = torch.Tensor(rots)
        gs_params['opacities'] = torch.Tensor(opacities)
        
        # Remove inf or nan
        select = torch.ones(gs_params['means'].shape[0], dtype=torch.bool)
        for key in gs_params:
            if key=='features_rest':
                select = select & ~torch.isnan(gs_params[key].sum(dim=1)).any(dim=1)
            else:
                select = select & ~torch.isnan(gs_params[key]).any(dim=1)
        for key in gs_params:
            gs_params[key] = gs_params[key][select]

        # Filter the outliers
        if self.remove_outlier_ndevs > 0:
            _,inlier_mask = remove_outliers(gs_params['means'], n_devs=self.remove_outlier_ndevs)
            for key in gs_params:
                gs_params[key] = gs_params[key][inlier_mask]

        # # Truncate gs params if num > self.max_gs_num
        # N = gs_params['means'].shape[0]
        # if N > self.max_gs_num:
        #     inlier_mask = torch.zeros(N, dtype=torch.bool)
        #     inlier_mask[:self.max_gs_num] = True
        #     for key in gs_params:
        #         gs_params[key] = gs_params[key][inlier_mask]

        # # Normalize the means and scales (we need to use the scaler to transform the camera later)
        # scaler = MinMaxScaler()
        # gs_params['means'] = scaler.fit_transform(gs_params['means']) 
        # gs_params['scales'] = gs_params['scales'] + torch.log(scaler.scale_)

        # inf_mask = torch.isinf(gs_params['scales']).sum(dim=1).bool()
        # valid_mask = (~inf_mask).bool()
        # inrange_mask = torch.all((gs_params['means'] >= 0) & (gs_params['means'] <= 1), dim=1)
        # valid_mask = valid_mask & inrange_mask
        # for key in gs_params:
        #     gs_params[key] = gs_params[key][valid_mask]
        #     if torch.isnan(gs_params[key]).any():
        #         print(f'Warning: {key} contains nan')

        scaler = None

        return gs_params, scaler


    def load_images_cameras_fromcolmap(self):
        sample_camera = self.train_caminfos[0]
        sample_image = Image.open(sample_camera.image_path)
        W, H = sample_image.size
        fov_x, fov_y = sample_camera.FovX, sample_camera.FovY
        fx, fy = fov2focal(fov_x, W), fov2focal(fov_y, H)
        cx, cy = W/2, H/2

        meta = {
            'fx':fx, 'fy':fy,
            'cx':cx, 'cy':cy,
            'width':W, 'height':H
        }

        for key in meta:
            meta[key] = torch.tensor(meta[key], dtype=torch.float32)

        train_c2ws, train_image_names = [], []
        test_c2ws, test_image_names = [], []
        for cam in self.train_caminfos:
            R, T = cam.R, cam.T
            w2c = getWorld2View(R, T)
            c2w = np.linalg.inv(w2c)
            c2w[0:3, 1:3] *= -1
            train_c2ws.append(c2w)
            train_image_names.append(cam.image_path)
        for cam in self.test_caminfos:
            R, T = cam.R, cam.T
            w2c = getWorld2View(R, T)
            c2w = np.linalg.inv(w2c)
            c2w[0:3, 1:3] *= -1
            test_c2ws.append(c2w)
            test_image_names.append(cam.image_path)
        trainposes = torch.from_numpy(np.array(train_c2ws).astype(np.float32))
        testposes = torch.from_numpy(np.array(test_c2ws).astype(np.float32))
        train_poses, test_poses = [], []
        train_imgs_path, test_imgs_path = [], []
        for i, name in enumerate(train_image_names):
            train_poses.append(trainposes[i])
            train_imgs_path.append(name)
        for i, name in enumerate(test_image_names):
            test_poses.append(testposes[i])
            test_imgs_path.append(name)
        if len(train_poses)!=0:
            meta['train_camera_to_worlds'] = torch.stack(train_poses, dim=0)
        else:
            print("Warning: No training images, set the 1st test poses as training poses as placeholder")
            meta['train_camera_to_worlds'] = torch.stack(test_poses[:1], dim=0)
            train_imgs_path = test_imgs_path[:1]
        meta['test_camera_to_worlds'] = torch.stack(test_poses, dim=0)
        return meta, train_imgs_path, test_imgs_path


    def load_scene(self, idx):
        ply_path = "/home/ZHANGAo_2024/gaussian-splatting/output/truck/sfm/point_cloud/iteration_30000/point_cloud.ply"
        # if os.path.exists(ply_path):
        #     gs_params, scaler = self.load_ply(ply_path)
        # else:
        #     gs_params, scaler = self.load_gs_params()
        gs_params, scaler = self.load_gs_params()
        meta, train_imgs_path, test_imgs_path = self.load_images_cameras_fromcolmap()
        # meta['train_camera_to_worlds'][:,:3,-1] = scaler.transform(meta['train_camera_to_worlds'][:,:3,-1])
        # meta['test_camera_to_worlds'][:,:3,-1] = scaler.transform(meta['test_camera_to_worlds'][:,:3,-1])

        outputs = {'gs_params': gs_params, 'meta': meta, 'idx': idx, 
                'scene_name': self.source_path.split('/')[-2], #The basename is splatfacto
                'train_imgs_path': train_imgs_path,
                'test_imgs_path': test_imgs_path}
        return outputs


    def get_scene_from_cache(self):
        return self.load_scene(1)


    def __iter__backup(self):
        gs_params_original, meta, scene_name = self.scene['gs_params'], self.scene['meta'], self.scene['scene_name']
        train_imgs_path, test_imgs_path = self.scene['train_imgs_path'], self.scene['test_imgs_path']
        train_imgs_name = [os.path.basename(path) for path in train_imgs_path]
        test_imgs_name = [os.path.basename(path) for path in test_imgs_path]

        N = gs_params_original['means'].shape[0]

        while True:
            total_train_num, total_test_num = len(meta['train_camera_to_worlds']), len(meta['test_camera_to_worlds'])
            cameras = {}
            if self.train_or_test == 'train':
                sample_test = np.random.rand(self.image_per_scene) < self.sample_ratio_test
                sample_test_num = min(np.sum(sample_test), total_test_num) #previously here it's max
                sample_train_num = self.image_per_scene - sample_test_num
                sample_train_num = min(sample_train_num, total_train_num)
                images, images_names = [], []
                cameras['camera_to_worlds'] = []
                #decide background_color
                if self.background_color == 'random':
                    background = torch.rand(3)
                else:
                    background = torch.tensor(self.background_color)/255.
                if sample_train_num > 0: #TODO not enough training or test views
                    train_cam_ids = np.random.permutation(total_train_num)[:sample_train_num]
                    images.extend([self.read_image(train_imgs_path[i], background=background) for i in train_cam_ids])
                    images_names.extend([train_imgs_name[i] for i in train_cam_ids])
                    cameras['camera_to_worlds'].append(meta['train_camera_to_worlds'][train_cam_ids])
                if sample_test_num > 0:
                    test_cam_ids = np.random.permutation(total_test_num)[:sample_test_num]
                    images.extend([self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids])
                    images_names.extend([test_imgs_name[i] for i in test_cam_ids])
                    cameras['camera_to_worlds'].append(meta['test_camera_to_worlds'][test_cam_ids])
                cameras['camera_to_worlds'] = torch.concatenate(cameras['camera_to_worlds'], axis=0)
            elif self.train_or_test == 'test':
                assert self.background_color!='random', 'For test set, background_color cannot be random'
                background = torch.tensor(self.background_color)/255.
                test_cam_ids = np.arange(total_test_num) #We take all the test images
                images = [self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids]
                images_names = [test_imgs_name[i] for i in test_cam_ids]
                cameras['camera_to_worlds'] = meta['test_camera_to_worlds'][test_cam_ids] 
            else:
                raise ValueError
            for key in ['fx','fy','cx','cy','width','height']:
                cameras[key] = meta[key]
            cameras['background_color'] = background
            ############ NOTE: sample gaussians ###########
            gs_params = {}
            if N > self.max_gs_num:
                mask = np.random.permutation(N)[:self.max_gs_num]
                for key in gs_params_original:
                    gs_params[key] = gs_params_original[key][mask]
            ###############################################
            # gs_param = gs_params_original
            ###############################################
            output_dict = {'gs_params': gs_params, 'images': images, 'cameras': cameras, 
                        'scene_idx': self.scene['idx'],
                        'scene_name': scene_name} #..../XX/splatfacto
            output_dict['images_name'] = images_names
            yield output_dict
            

    def __iter__(self):
        gs_params_original, meta, scene_name = self.scene['gs_params'], self.scene['meta'], self.scene['scene_name']
        train_imgs_path, test_imgs_path = self.scene['train_imgs_path'], self.scene['test_imgs_path']
        train_imgs_name = [os.path.basename(path) for path in train_imgs_path]
        test_imgs_name = [os.path.basename(path) for path in test_imgs_path]

        total_train_num, total_test_num = len(meta['train_camera_to_worlds']), len(meta['test_camera_to_worlds'])
        ### NOTE: trying ###
        # total_train_num = 9
        ####################
        training_index = np.random.permutation(total_train_num)
        N = gs_params_original['means'].shape[0]

        ############ NOTE: fixed sample gaussians ###########
        # gs_params = {}
        # if N > self.max_gs_num:
        #     inlier_mask = torch.zeros(N, dtype=torch.bool)
        #     inlier_mask[:self.max_gs_num] = True
        #     for key in gs_params_original:
        #         gs_params[key] = gs_params_original[key][inlier_mask]
        #####################################################

        while True:
            training_index = np.random.permutation(total_train_num)
            for i in range(0, len(training_index), self.image_per_scene):
                train_cam_ids = training_index[i:i+self.image_per_scene]
                cameras = {}
                if self.train_or_test == 'train':
                    images, images_names = [], []
                    cameras['camera_to_worlds'] = []
                    #decide background_color
                    if self.background_color == 'random':
                        background = torch.rand(3)
                    else:
                        background = torch.tensor(self.background_color)/255.
                    images.extend([self.read_image(train_imgs_path[i], background=background) for i in train_cam_ids])
                    images_names.extend([train_imgs_name[i] for i in train_cam_ids])
                    cameras['camera_to_worlds'].append(meta['train_camera_to_worlds'][train_cam_ids])
                    cameras['camera_to_worlds'] = torch.concatenate(cameras['camera_to_worlds'], axis=0)
                elif self.train_or_test == 'test':
                    assert self.background_color!='random', 'For test set, background_color cannot be random'
                    background = torch.tensor(self.background_color)/255.
                    test_cam_ids = np.arange(total_test_num) #We take all the test images
                    images = [self.read_image(test_imgs_path[i], background=background) for i in test_cam_ids]
                    images_names = [test_imgs_name[i] for i in test_cam_ids]
                    cameras['camera_to_worlds'] = meta['test_camera_to_worlds'][test_cam_ids] 
                else:
                    raise ValueError
                for key in ['fx','fy','cx','cy','width','height']:
                    cameras[key] = meta[key]
                cameras['background_color'] = background
                ############ NOTE: sample gaussians ###########
                # gs_params = {}
                # if N > self.max_gs_num:
                #     mask = np.random.permutation(N)[:self.max_gs_num]
                #     for key in gs_params_original:
                #         gs_params[key] = gs_params_original[key][mask]
                ###############################################
                gs_params = gs_params_original
                ###############################################
                output_dict = {'gs_params': gs_params, 
                               'images': images, 'cameras': cameras, 
                               'scene_idx': self.scene['idx'], 
                               'scene_name': scene_name} #..../XX/splatfacto
                output_dict['images_name'] = images_names
                yield output_dict
                