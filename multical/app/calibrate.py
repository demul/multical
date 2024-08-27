
from multical.io.logging import setup_logging
from .vis import visualize_ws

from structs.struct import struct, map_none, to_structs
import numpy as np

from multical.config import *
import multical.tables
from dataclasses import dataclass

@dataclass
class Calibrate:
    """Run camera calibration"""
    paths  : PathOpts 
    camera  : CameraOpts
    runtime    : RuntimeOpts
    optimizer  : OptimizerOpts 
    vis : bool = False        # Visualize result after calibration

    def execute(self):
        calibrate(self)


def calibrate(args): 
  np.set_printoptions(precision=4, suppress=True)

  # Use image path if not explicity specified
  output_path = args.paths.image_path or args.paths.output_path 

  ws = workspace.Workspace(output_path, args.paths.name)
  setup_logging(args.runtime.log_level, [ws.log_handler], log_file=path.join(output_path, f"{args.paths.name}.txt"))

  boards = find_board_config(args.paths.image_path, board_file=args.paths.boards)
  camera_images = find_camera_images(
    args.paths.image_path, 
    args.paths.cameras, 
    args.paths.camera_pattern, 
    limit=9999999
  )
  ws.add_camera_images(camera_images, j=args.runtime.num_theads)
  ws.detect_boards(boards, load_cache=False, j=args.runtime.num_theads)
  full_detected_points = ws.detected_points

  calibration_json_dict = None
  for camera_count_to_calibrate in range(2, len(args.paths.cameras) + 1):
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
        loss='linear', 
        outlier_quantile=0.75, 
        outlier_threshold=25.0, 
        auto_scale=None, 
        fix_intrinsic=False, 
        fix_camera_poses=False, 
        fix_board_poses=False, 
        fix_motion=False, 
        adjust_board=False
        ),
        "initialisation"
      )
    calibration_json_dict: dict = ws.export()
  ws.dump(calibration_json_dict)


if __name__ == '__main__':
  run_with(Calibrate)
