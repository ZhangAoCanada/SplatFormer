#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from .utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from .utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:
    def __init__(self, 
                model_path="outputs/truck/trying_debug",
                source_path="data/tandt_db/tandt/truck",
                load_iteration=None, 
                shuffle=True, 
                resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = model_path
        self.loaded_iter = None
        self.white_background = False
        self.eval = True

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, "images", self.eval)
        elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](source_path, self.white_background, self.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_camera_infos = scene_info.train_cameras
        self.test_camera_infos = scene_info.test_cameras
        self.point_cloud = scene_info.point_cloud
        self.cameras_extent = scene_info.nerf_normalization["radius"]

    
    def getTrainCameraInfos(self):
        return self.train_camera_infos
    
    def getTestCameraInfos(self):
        return self.test_camera_infos