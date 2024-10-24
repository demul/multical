import argparse
import sys

import numpy as np

from multical.io.logging import setup_logging
from multical.io.import_calib import load_calibration
from multical.config import *
from multical.optimization.calibration import Calibration
import multical.tables


def main(args):
    np.set_printoptions(precision=4, suppress=True)

    image_dir_path = args["image_dir_path"]
    board_config_file_path = args["board_config_file_path"]
    thread_count = int(args["thread_count"])
    camera_pairs = args["camera_pairs"]
    
    camera_names = []
    camera_name_to_camera_index_map = dict()
    for camera_pair_index, camera_pair in enumerate(camera_pairs):
        if camera_pair_index == 0:
            reference_camera_name = camera_pair.split("-")[0]
            reference_camera_index = 0 # reference camera index is always 0
            camera_names.append(reference_camera_name)
            camera_name_to_camera_index_map[reference_camera_name] = reference_camera_index
        camera_name = camera_pair.split("-")[1]
        camera_index = camera_pair_index + 1
        camera_names.append(camera_name)
        camera_name_to_camera_index_map[camera_name] = camera_index
        
    camera_index_pairs = []
    for camera_pair in camera_pairs:
        camera_name_pair = camera_pair.split("-")
        camera_index_pairs.append((camera_name_to_camera_index_map[camera_name_pair[0]], camera_name_to_camera_index_map[camera_name_pair[1]]))
    
    ws = workspace.Workspace(image_dir_path, "calibration")
    setup_logging(
        "INFO",
        [ws.log_handler],
        log_file=path.join(image_dir_path, "calibration.txt"),
    )

    boards = find_board_config(image_dir_path, board_file=board_config_file_path)
    camera_images = find_camera_images(
        image_dir_path,
        camera_names,
        None,
        limit=9999999,
    )
    ws.add_camera_images(camera_images, j=thread_count)
    ws.detect_boards(boards, load_cache=False, j=thread_count)
    full_detected_points = ws.detected_points
    full_image_sizes = ws.image_size
    
    calibration_json_dict = None
    for camera_count_to_calibrate in range(2, len(camera_names) + 1):
        ws.detected_points = full_detected_points[:camera_count_to_calibrate]
        ws.image_size = full_image_sizes[:camera_count_to_calibrate]
        ws.point_table = multical.tables.make_point_table(ws.detected_points, ws.boards)
        initialise_with_images(
            ws,
            CameraOpts(
                fix_aspect=False,
                allow_skew=False,
                distortion_model="standard",
                motion_model="static",
                isFisheye=False,
                calibration=calibration_json_dict,
                limit_intrinsic=9999999,
            ),
            camera_index_pairs,
        )
        optimize(
            ws,
            OptimizerOpts(
                iter=10, 
                loss='linear', 
                outlier_quantile=0.75, 
                outlier_threshold=25.0, 
                auto_scale=None, 
                fix_intrinsic=False, 
                fix_camera_poses=False, 
                fix_board_poses=False, 
                fix_motion=False, 
                adjust_board=False),
                "initialisation"
            )
        calibration_json_dict: dict = ws.export()
    ws.dump(calibration_json_dict)


def get_args():
    parser = argparse.ArgumentParser("Camera Calibration")
    parser.add_argument(
        "--image_dir_path", 
        "-i", 
        type=str, 
        help="Path of image directory. \
            \nex) -i /media/link/data/calibration", 
        required=True
    )
    parser.add_argument(
        "--camera_pairs", 
        "-c", 
        nargs="+", 
        help="List of camera pairs to calibrate. Rig transformation is calibrated incrementally in the input order. \
            \nex) -c cam_f-cam_fr cam_fr-cam_br cam_br-cam_bl cam_bl-cam_fl cam_f-depth_fl_infra2 depth_fl_infra2-depth_fl_infra1 cam_f-depth_fr_infra1 depth_fr_infra1-depth_fr_infra2", 
        required=True
    )
    parser.add_argument(
        "--thread_count", 
        "-j", 
        type=str,
        help="Thread count using for calibration. \
            \ndefault) -j 8",
        default=8
    )
    parser.add_argument(
        "--board_config_file_path", 
        "-b", 
        type=str,
        help="Path of borad config file. \
            \ndefault) -b /root/autonomy-camera-calibration/example_boards/charuco_5x5.yaml",
        default="/root/autonomy-camera-calibration/example_boards/charuco_5x5.yaml"
    )
    args = parser.parse_args()
    args = vars(args)  # convert to dictionary
    return args


if __name__ == "__main__":
    args_ = get_args()
    main(args_)
