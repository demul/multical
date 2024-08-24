import argparse
import sys

import numpy as np

from multical.io.logging import setup_logging
from multical.io.import_calib import load_calibration
from multical.config import *
from multical.optimization.calibration import Calibration
import multical.tables

# sys.tracebacklimit = 0


def main(args):
    np.set_printoptions(precision=4, suppress=True)

    image_dir_path = args["image_dir_path"]
    board_config_file_path = args["board_config_file_path"]
    thread_count = int(args["thread_count"])

    ordered_camera_name_list = ["cam_f", "cam_fr", "cam_br", "cam_bl", "cam_fl"]

    ws = workspace.Workspace(image_dir_path, "calibration")
    setup_logging(
        "INFO",
        [ws.log_handler],
        log_file=path.join(image_dir_path, "calibration.txt"),
    )

    boards = find_board_config(image_dir_path, board_file=board_config_file_path)
    camera_images = find_camera_images(
        image_dir_path,
        ordered_camera_name_list,
        None,
        limit=9999999,
    )
    ws.add_camera_images(camera_images, j=thread_count)
    ws.detect_boards(boards, load_cache=False, j=thread_count)
    full_detected_points = ws.detected_points

    calibration_json_dict = None
    for camera_count_to_calibrate in range(2, len(ordered_camera_name_list) + 1):
        ws.detected_points = full_detected_points[:camera_count_to_calibrate]
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
        )
        optimize(
            ws,
            OptimizerOpts(
                iter=10,
                loss="linear",
                outlier_quantile=0.75,
                outlier_threshold=25.0,
                auto_scale=None,
                fix_intrinsic=False,
                fix_camera_poses=False,
                fix_board_poses=False,
                fix_motion=False,
                adjust_board=False,
            ),
            "initialisation",
        )
        calibration_json_dict: dict = ws.export()


def get_args():
    parser = argparse.ArgumentParser("Camera Calibration")
    parser.add_argument("--image_dir_path", "-i", type=str)
    parser.add_argument("--thread_count", "-j", type=str, default=8)
    parser.add_argument(
        "--board_config_file_path", "-b", type=str, default="/root/charuco_5x5.yaml"
    )
    args = parser.parse_args()
    args = vars(args)  # convert to dictionary
    return args


if __name__ == "__main__":
    args_ = get_args()
    main(args_)
