import argparse
import sys

import numpy as np

from multical.io.logging import setup_logging
from multical.config import *

sys.tracebacklimit = 0


def main(args):
    np.set_printoptions(precision=4, suppress=True)

    image_dir_path = args["image_dir_path"]
    board_config_file_path = args["board_config_file_path"]
    thread_count = int(args["thread_count"])

    ws = workspace.Workspace(image_dir_path, "calibration")
    setup_logging(
        "INFO",
        [ws.log_handler],
        log_file=path.join(image_dir_path, "calibration.txt"),
    )

    boards = find_board_config(image_dir_path, board_file=board_config_file_path)
    camera_images = find_camera_images(
        image_dir_path,
        ["cam_f", "cam_fr", "cam_br", "cam_bl", "cam_fl"],
        None,
        limit=9999999,
    )

    initialise_with_images(
        ws,
        boards,
        camera_images,
        CameraOpts(
            fix_aspect=False,
            allow_skew=False,
            distortion_model="standard",
            motion_model="static",
            isFisheye=False,
            calibration=None,
            limit_intrinsic=9999999,
        ),
        RuntimeOpts(num_threads=thread_count, log_level="INFO", no_cache=False, seed=0),
    )
    optimize(ws, args.optimizer)

    json_dict: dict = ws.export()


def get_args():
    parser = argparse.ArgumentParser("Camera Calibration for Samsung AEBT")
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
