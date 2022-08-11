import argparse
import copy
import glob
import os
import random
import shutil
import json

import cv2
import numpy as np


class Camera:
    def __init__(self, intrinsic, distortion, rig_transformation):
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.rig_transformation = rig_transformation


class CameraCalibrationScriptRunner:
    def __init__(self, calibration_images_dir_path):
        self.calibration_images_dir_path = calibration_images_dir_path
        self.input_camera_dir_path_list = self.get_camera_dir_path_list()
        self.camera_name_list = [os.path.basename(camera_dir_path) for camera_dir_path in self.input_camera_dir_path_list]
        self.camera_index_list = [int(camera_name[3:]) for camera_name in self.camera_name_list]
        self.maximum_image_index = self.get_maximum_image_index()

    def get_camera_dir_path_list(self):
        input_camera_dir_path_list = glob.glob(os.path.join(self.calibration_images_dir_path, "cam*"))
        input_camera_dir_path_list = [camera_dir_path for camera_dir_path in input_camera_dir_path_list if os.path.isdir(camera_dir_path)]
        return input_camera_dir_path_list

    def get_maximum_image_index(self):
        maximum_image_indices_of_each_subdir = []
        for camera_index in self.camera_index_list:
            image_filename_list = os.listdir(os.path.join(self.calibration_images_dir_path, "cam%d" % camera_index))
            image_filename_without_extension_list = [int(image_filename.split(".")[0]) for image_filename in image_filename_list]
            maximum_image_index_of_this_subdir = max(image_filename_without_extension_list)
            maximum_image_indices_of_each_subdir.append(maximum_image_index_of_this_subdir)
        return max(maximum_image_indices_of_each_subdir)

    def shuffle_image_indices(self, output_dir_path):
        output_camera_dir_path_list = [os.path.join(output_dir_path, camera_name) for camera_name in self.camera_name_list]
        image_file_name_list = ["%d.png" % image_index for image_index in range(self.maximum_image_index)]

        from_name_list = copy.deepcopy(image_file_name_list)
        to_name_list = copy.deepcopy(image_file_name_list)
        random.shuffle(to_name_list)

        for input_camera_dir_path, output_camera_dir_path in zip(self.input_camera_dir_path_list, output_camera_dir_path_list):
            os.makedirs(output_camera_dir_path)
            for image_index in range(self.maximum_image_index):
                from_path = os.path.join(input_camera_dir_path, from_name_list[image_index])
                to_path = os.path.join(output_camera_dir_path, to_name_list[image_index])
                if not os.path.exists(from_path):
                    continue
                shutil.copy(from_path, to_path)

    def run_multical(self, input_calibration_images_dir_path):
        camera_name_enum_str = ""
        for camera_name in self.camera_name_list:
            camera_name_enum_str += camera_name + " "
        camera_name_enum_str = camera_name_enum_str[:-1]

        os.system("multical calibrate --cameras " + camera_name_enum_str + " --image_path " + input_calibration_images_dir_path + " --board /home/multical/example_boards/aprilgrid_3x3_multiple.yaml --limit_intrinsic 1000 --limit_images 1000 --num_thread 16")


    def get_camera_list_from_json(self, json_file_path):
        with open(json_file_path, "r") as fd:
            json_dict = json.load(fd)
        camera_list = [
            self.get_camera_from_json_dict(json_dict, camera_index)
            for camera_index in self.camera_index_list
        ]
        return camera_list

    def get_camera_from_json_dict(self, json_dict, camera_index):
        intrinsic_dict = json_dict["cameras"]
        rig_transformation_dict = json_dict["camera_poses"]
        if "cam%d" % camera_index in intrinsic_dict.keys():
            intrinsic = np.array(
                intrinsic_dict["cam%d" % camera_index]["K"], dtype=np.float64
            )
            distortion = np.array(
                intrinsic_dict["cam%d" % camera_index]["dist"], dtype=np.float64
            )
        else:
            intrinsic = None
            distortion = None

        if camera_index is 0:
            if "cam0" in rig_transformation_dict.keys():
                extrinsic = np.empty((4, 4), dtype=np.float64)
                rotation = np.array(rig_transformation_dict["cam0"]["R"], dtype=np.float64)
                translation = np.array(
                    rig_transformation_dict["cam0"]["T"], dtype=np.float64
                )
                extrinsic[:3, :3] = rotation
                extrinsic[:3, 3] = translation
                extrinsic[3, :] = [0, 0, 0, 1]
                rig_transformation = np.linalg.inv(extrinsic)
            else:
                rig_transformation = None
        else:
            if "cam%d_to_cam0" % camera_index in rig_transformation_dict.keys():
                extrinsic = np.empty((4, 4), dtype=np.float64)
                rotation = np.array(
                    rig_transformation_dict["cam%d_to_cam0" % camera_index]["R"],
                    dtype=np.float64,
                )
                translation = np.array(
                    rig_transformation_dict["cam%d_to_cam0" % camera_index]["T"],
                    dtype=np.float64,
                )
                extrinsic[:3, :3] = rotation
                extrinsic[:3, 3] = translation
                extrinsic[3, :] = [0, 0, 0, 1]
                rig_transformation = np.linalg.inv(extrinsic)
            else:
                rig_transformation = None
        return Camera(intrinsic, distortion, rig_transformation)

    def factorize_intrinsic_matrix(self, intrinsic_matrix):
        return (
            float(intrinsic_matrix[0, 0]),
            float(intrinsic_matrix[1, 1]),
            float(intrinsic_matrix[0, 2]),
            float(intrinsic_matrix[1, 2]),
        )

    def save_camera_to_yaml(self, fd, camera, camera_index):
        if camera.intrinsic is not None:
            fx, fy, cx, cy = self.factorize_intrinsic_matrix(camera.intrinsic)
            fd.startWriteStruct("cam%d_parameters" % camera_index, cv2.FileNode_MAP)
            fd.write("fx", fx)
            fd.write("fy", fy)
            fd.write("cx", cx)
            fd.write("cy", cy)
            fd.endWriteStruct()

        if camera.distortion is not None:
            fd.write("cam%d_distortions" % camera_index, camera.distortion)

        if camera.rig_transformation is not None:
            fd.write("cam%dRotation" % camera_index, camera.rig_transformation[:3, :3])
            fd.write("cam%dTranslation" % camera_index, camera.rig_transformation[:3, 3])

    def save_cameras_to_yaml(self, yaml_file_path, camera_list):
        fd = cv2.FileStorage(yaml_file_path, cv2.FILE_STORAGE_WRITE)
        for camera, camera_index in zip(camera_list, self.camera_index_list):
            self.save_camera_to_yaml(fd, camera, camera_index)
        fd.release()

    def convert_camera_parameter_format(self, input_json_path, output_yaml_path):
        camera_list = self.get_camera_list_from_json(input_json_path)
        self.save_cameras_to_yaml(output_yaml_path, camera_list)


def get_args():
    parser = argparse.ArgumentParser("Multical Camera Calibration")
    parser.add_argument(
        "--input_calibration_dir_path",
        "-i",
        type=str,
    )
    parser.add_argument(
        "--output_calibration_dir_path",
        "-o",
        type=str,
    )
    args = parser.parse_args()
    args = vars(args)
    return args


if __name__ == "__main__":
    args_ = get_args()
    script_runner = CameraCalibrationScriptRunner(args_["input_calibration_dir_path"])
    script_runner.shuffle_image_indices(args_["output_calibration_dir_path"])
    script_runner.run_multical(args_["output_calibration_dir_path"])
    script_runner.convert_camera_parameter_format(
        os.path.join(args_["output_calibration_dir_path"], 'calibration.json'),
        os.path.join(args_["output_calibration_dir_path"], 'camera_params.yaml')
    )

    