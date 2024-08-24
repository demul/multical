import cv2

from collections import OrderedDict
from multical.threading import parmap_lists
import pathlib
from multical.board.board import Board

import numpy as np
from multical.motion import StaticFrames
from multiprocessing import cpu_count

from multical.optimization.parameters import ParamList
from multical.optimization.pose_set import PoseSet
from multical import config

from os import path
from multical.io import export_json, try_load_detections, write_detections
from multical.image.detect import common_image_size

from multical.optimization.calibration import Calibration, select_threshold
from structs.struct import map_list, split_dict, struct, subset, to_dicts
from . import tables, image
from .camera import calibrate_cameras

from structs.numpy import shape

from .camera_fisheye import calibrate_cameras_fisheye
from .io.logging import MemoryHandler, info
from .display import color_sets

import pickle
import json


class NbCamera:
    def __init__(
        self,
        camera_name,
        width: int = None,
        height: int = None,
        intrinsic: np.ndarray = None,
        distortion: dict = None,
        rig_transformation: np.ndarray = None,
    ):
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.rig_transformation = rig_transformation


def read_camera_from_multical_json_dict(
    json_dict: dict,
    reference_camera_name: str,
    target_camera_name: str,
    distortion_parameter_name_order: list,
) -> NbCamera:
    intrinsic_dict = json_dict["cameras"]
    rig_transformation_dict = json_dict["camera_poses"]

    if target_camera_name in intrinsic_dict.keys():
        width = int(intrinsic_dict[target_camera_name]["image_size"][0])
        height = int(intrinsic_dict[target_camera_name]["image_size"][1])
        intrinsic = np.array(intrinsic_dict[target_camera_name]["K"], dtype=np.float64)
        distortion = dict()
        for distortion_parameter_name, distortion_parameter_value in zip(
            distortion_parameter_name_order,
            intrinsic_dict[target_camera_name]["dist"][0],
        ):
            distortion[distortion_parameter_name] = distortion_parameter_value

    else:
        print("[%s] intrinsic and distortion does not exist!" % target_camera_name)
        width = None
        height = None
        intrinsic = None
        distortion = None

    if target_camera_name == reference_camera_name:
        key_str = target_camera_name
    else:
        key_str = "%s_to_%s" % (target_camera_name, reference_camera_name)

    if key_str in rig_transformation_dict.keys():
        extrinsic = np.empty((4, 4), dtype=np.float64)
        rotation = np.array(rig_transformation_dict[key_str]["R"], dtype=np.float64)
        translation = np.array(rig_transformation_dict[key_str]["T"], dtype=np.float64)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        extrinsic[3, :] = [0, 0, 0, 1]
        rig_transformation = np.linalg.inv(extrinsic)
    else:
        print("[%s] rig transformation does not exist!" % target_camera_name)
        rig_transformation = None

    return NbCamera(
        camera_name=target_camera_name,
        width=width,
        height=height,
        intrinsic=intrinsic,
        distortion=distortion,
        rig_transformation=rig_transformation,
    )


def read_cameras_from_multical_json(
    json_dict: dict,
    camera_name_list: list,
    reference_camera_name: str,
    distortion_parameter_name_order: list,
) -> list:
    camera_list = [
        read_camera_from_multical_json_dict(
            json_dict,
            reference_camera_name,
            target_camera_name,
            distortion_parameter_name_order,
        )
        for target_camera_name in camera_name_list
    ]
    return camera_list


def factorize_intrinsic_matrix(
    intrinsic_matrix: np.ndarray,
):
    return (
        float(intrinsic_matrix[0, 0]),
        float(intrinsic_matrix[1, 1]),
        float(intrinsic_matrix[0, 2]),
        float(intrinsic_matrix[1, 2]),
    )


def write_camera_as_neubility_yaml_version_2(fd: cv2.FileStorage, camera: NbCamera):
    fd.startWriteStruct(camera.camera_name, cv2.FileNode_MAP)

    if camera.width is not None:
        fd.write("width", camera.width)

    if camera.height is not None:
        fd.write("height", camera.height)

    if camera.intrinsic is not None:
        fd.startWriteStruct("intrinsic", cv2.FileNode_MAP)
        fx, fy, cx, cy = factorize_intrinsic_matrix(camera.intrinsic)
        fd.write("fx", fx)
        fd.write("fy", fy)
        fd.write("cx", cx)
        fd.write("cy", cy)
        fd.endWriteStruct()

    if camera.distortion is not None:
        fd.startWriteStruct("distortion", cv2.FileNode_MAP)
        for (
            distortion_parameter_name,
            distortion_parameter_value,
        ) in camera.distortion.items():
            fd.write(distortion_parameter_name, distortion_parameter_value)
        fd.endWriteStruct()

    if camera.rig_transformation is not None:
        fd.write("rig_transformation", camera.rig_transformation)
    fd.endWriteStruct()


def write_cameras_as_neubility_yaml(
    yaml_file_path: str, camera_list: list, version: int
):
    fd = cv2.FileStorage(yaml_file_path, cv2.FILE_STORAGE_WRITE)
    fd.write("version", version)
    for camera in camera_list:
        write_camera_as_neubility_yaml_version_2(fd, camera)
    fd.release()


def detect_boards_cached(boards, images, detections_file, cache_key, load_cache=True, j=cpu_count()):
  assert isinstance(boards, list)

  detected_points = (try_load_detections(
    detections_file, cache_key) if load_cache else None)

  if detected_points is None:
    info("Detecting boards..")
    detected_points = image.detect.detect_images(boards, images, j=j)

    info(f"Writing detection cache to {detections_file}")
    write_detections(detections_file, detected_points, cache_key)

  return detected_points

def num_valid_detections(boards, frames):
  n = 0
  for frame_detections in frames:
    for board, dets in zip(boards, frame_detections):
      if board.has_min_detections(dets): n = n + 1
  return n

def check_detections(camera_names, boards, detected_points):
  cameras = [k for k, fame_detections in zip(camera_names, detected_points)
    if num_valid_detections(boards, fame_detections) == 0]
    
  assert len(cameras) == 0,\
    f"cameras {cameras} have no valid detections, check board config"
    

def check_image_lengths(cameras, filenames, image_names):
  for k, images in zip(cameras, filenames):
    assert len(images) == len(image_names),\
      f"mismatch between image names and camera {k}, "\
      f"got {len(images)} filenames expected {len(image_names)}"


def check_camera_images(camera_images):
  assert len(camera_images.cameras) == len(camera_images.filenames),\
    f"expected filenames to be a list of equal to number of cameras "\
    f"{len(camera_images.cameras)} vs. {len(camera_images.filenames)}"

  check_image_lengths(camera_images.cameras, camera_images.filenames, camera_images.image_names)
  if 'images' in camera_images is not None:
    check_image_lengths(camera_images.cameras, camera_images.images, camera_images.image_names)



class Workspace:

    def __init__(self, output_path, name="calibration"):

        self.name = name
        self.output_path = output_path

        self.calibrations = OrderedDict()
        self.detections = None
        self.boards = None
        self.board_colors = None

        self.filenames = None
        self.image_path = None
        self.names = struct()

        self.image_sizes = None
        self.images = None

        self.point_table = None
        self.pose_table = None

        self.log_handler = MemoryHandler()

    def add_camera_images(self, camera_images, j=cpu_count()):
        check_camera_images(camera_images)
        self.names = self.names._extend(
            camera=camera_images.cameras, image=camera_images.image_names)

        self.filenames = camera_images.filenames
        self.image_path = camera_images.image_path
        
        if 'images' in camera_images:
          self.images = camera_images.images
          self.image_size = map_list(common_image_size, self.images)
        else:
          self._load_images(j=j)

    def _load_images(self, j=cpu_count()):
        assert self.filenames is not None, "_load_images: no filenames set"

        info("Loading images..")
        self.images = image.detect.load_images(
            self.filenames, j=j, prefix=self.image_path)
        self.image_size = map_list(common_image_size, self.images)

        info(f"Loaded {self.sizes.image * self.sizes.camera} images")
        info(
            {k: image_size for k, image_size in zip(
                self.names.camera, self.image_size)})

              
    @property 
    def detections_file(self):
      return path.join(self.output_path, f"{self.name}.detections.pkl")


    def detect_boards(self, boards, load_cache=True, j=cpu_count()):
        assert self.boards is None, "detect_boards: boards already set"
        assert self.images is not None, "detect_boards: no images loaded, first use add_camera_images"

        board_names, self.boards = split_dict(boards)
        self.names = self.names._extend(board=board_names)
        self.board_colors = color_sets['set1']
        cache_key = self.fields("filenames", "boards", "image_sizes")

        self.detected_points = detect_boards_cached(self.boards, self.images, 
          self.detections_file, cache_key, load_cache, j=j)

        self.point_table = tables.make_point_table(self.detected_points, self.boards)
        info("Detected point counts:")
        tables.table_info(self.point_table.valid, self.names)

    def set_calibration(self, cameras):
      self.cameras = list(self.cameras)
      for camera_index, camera_name in enumerate(self.names.camera):
        if camera_name in cameras.keys():
          self.cameras[camera_index] = cameras[camera_name]
      
      info("Cameras set...")
      for name, camera in zip(self.names.camera, self.cameras):
          info(f"{name} {camera}")
          info("")

    def calibrate_single(self, camera_model, fix_aspect=False, has_skew=False, max_images=None, isFisheye=False):
        assert self.detected_points is not None, "calibrate_single: no points found, first use detect_boards to find corner points"

        check_detections(self.names.camera, self.boards, self.detected_points)

        info("Calibrating single cameras..")
        if not isFisheye:
            self.cameras, errs = calibrate_cameras(
                self.boards,
                self.detected_points,
                self.image_size,
                model=camera_model,
                fix_aspect=fix_aspect,
                has_skew=has_skew,
                max_images=max_images)
        else:
            self.cameras, errs = calibrate_cameras_fisheye(
                self.boards,
                self.detected_points,
                self.image_size,
                model=camera_model,
                fix_aspect=fix_aspect,
                has_skew=has_skew,
                max_images=max_images)

        for name, camera, err in zip(self.names.camera, self.cameras, errs):
            info(f"Calibrated {name}, with RMS={err:.2f}")
            info(camera)
            info("")

    def initialise_poses(self, motion_model=StaticFrames, camera_poses=None, isFisheye=False):
        assert self.cameras is not None, "initialise_poses: no cameras set, first use calibrate_single or set_cameras"
        self.pose_table = tables.make_pose_table(self.point_table, self.boards, self.cameras)

        info("Pose counts:")
        tables.table_info(self.pose_table.valid, self.names)

        if camera_poses is not None:
          camera_poses_npy = np.empty([len(camera_poses.keys()), 4, 4], dtype=next(iter(camera_poses.values())).dtype)
          for camera_index, camera_name in enumerate(self.names.camera):
            if camera_name not in camera_poses.keys():
              continue
            camera_poses_npy[camera_index] = camera_poses[camera_name]
        else:
          camera_poses_npy = None

        pose_init = tables.initialise_poses(self.pose_table, 
          camera_poses=camera_poses_npy
        )

        calib = Calibration(
            ParamList(self.cameras, self.names.camera),
            ParamList(self.boards, self.names.board),
            self.point_table,
            PoseSet(pose_init.camera, self.names.camera),
            PoseSet(pose_init.board, self.names.board),
            motion_model.init(pose_init.times, self.names.image),
        )

        # calib = calib.reject_outliers_quantile(0.75, 5)
        calib.report(f"Initialisation")

        self.calibrations["initialisation"] = calib
        return calib

    def calibrate(self, name="calibration",
        camera_poses=True, motion=True, board_poses=True, 
        cameras=False, boards=False,
        loss='linear', tolerance=1e-4, num_adjustments=5,
        quantile=0.75, auto_scale=None, outlier_threshold=5.0)  -> Calibration:

        calib : Calibration = self.latest_calibration.enable(
            cameras=cameras, boards=boards, camera_poses=camera_poses,
            motion=motion, board_poses=board_poses)
            
        calib = calib.adjust_outliers(
          loss=loss, 
          tolerance=tolerance,
          num_adjustments=num_adjustments,
          select_outliers = select_threshold(quantile=quantile, factor=outlier_threshold),
          select_scale = select_threshold(quantile=quantile, factor=auto_scale) if auto_scale is not None else None
        )

        self.calibrations[name] = calib
        return calib

    @property
    def sizes(self):
        return self.names._map(len)

    @property
    def initialisation(self)  -> Calibration:
        return self.calibrations["initialisation"]

    @property
    def latest_calibration(self) -> Calibration:
        return list(self.calibrations.values())[-1]

    @property
    def log_entries(self):
        return self.log_handler.records

    def has_calibrations(self):
        return len(self.calibrations) > 0

    def get_calibrations(self):
        return self.calibrations

    def push_calibration(self, name, calib):
      if name in self.calibrations:
        raise KeyError(f"calibration {name} exists already {list(self.calibrations.keys())}")
      self.calibrations[name] = calib

    def get_camera_sets(self):
        if self.has_calibrations():
            return {k: calib.cameras for k, calib in self.calibrations.items()}

        if self.cameras is not None:
            return dict(initialisation=self.cameras)

    def export_json(self, master=None):
        master = master or self.names.camera[0]
        assert (
            master is None or master in self.names.camera
        ), f"master f{master} not found in cameras f{str(self.names.camera)}"

        calib = self.latest_calibration
        if master is not None:
            calib = calib.with_master(master)

        return export_json(calib, self.names, self.filenames, master=master)


    def export(self, filename=None, master=None):
        data = self.export_json(master=master)
        return to_dicts(data)
      
    def save_cameras_as_neubility_yaml(self, json_dict: dict):
        filename = path.join(self.output_path, f"{self.name}.yaml")
        camera_list = read_cameras_from_multical_json(
            json_dict,
            camera_name_list=["cam_f", "cam_fr", "cam_bl", "cam_br", "cam_fl"],
            reference_camera_name="cam_f",
            distortion_parameter_name_order=["k1", "k2", "r1", "r2", "k3"],
        )
        base_link = NbCamera(
            camera_name="baselink_projected_ground",
            rig_transformation=np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.339],
                    [1.0, 0.0, 0.0, -0.254],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        base_link_projected_ground = NbCamera(
            camera_name="baselink_projected_ground",
            rig_transformation=np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.561],
                    [1.0, 0.0, 0.0, -0.254],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        camera_list.append(base_link)
        camera_list.append(base_link_projected_ground)
        write_cameras_as_neubility_yaml(
            yaml_file_path=filename,
            camera_list=camera_list,
            version=2,
        )
        
    def dump(self, filename=None):
        filename = filename or path.join(self.output_path, f"{self.name}.pkl")

        info(f"Dumping state and history to {filename}")
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        assert path.isfile(
            filename), f"Workspace.load: file does not exist {filename}"
        with open(filename, "rb") as file:
            ws = pickle.load(file)
            return ws

    def fields(self, *keys):
        return subset(self.__dict__, keys)

    def __getstate__(self):
        return self.fields(
            "calibrations",
            "detections",
            "boards",
            "board_colors",
            "filenames",
            "image_path",
            "names",
            "image_sizes",
            "point_table",
            "pose_table",
            "log_handler",
        )

    def __setstate__(self, d):
        for k, v in d.items():
            self.__dict__[k] = v

        self.images = None
