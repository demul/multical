from structs.struct import map_none
from multical.io.import_calib import load_calibration
from multical.motion.static_frames import StaticFrames
from multical.motion.rolling_frames import RollingFrames
from multical.workspace import Workspace

from .arguments import *


def get_motion_model(motion_model):
    if motion_model == "rolling":
        return RollingFrames
    elif motion_model == "static":
        return StaticFrames
    else:
        assert False, f"unknown motion model {motion_model}, (static|rolling)"


def initialise_with_images(
    ws: Workspace,
    camera_opts: CameraOpts = CameraOpts(),
    camera_index_pairs: list = None,
):
    calib = map_none(load_calibration, camera_opts.calibration)
    
    if calib is None:
        ws.calibrate_single(
            camera_opts.distortion_model,
            ws.detected_points,
            ws.image_size,
            fix_aspect=camera_opts.fix_aspect,
            has_skew=camera_opts.allow_skew,
            max_images=camera_opts.limit_intrinsic,
            isFisheye=camera_opts.isFisheye,
        )
    else:
        ws.calibrate_single(
            camera_opts.distortion_model,
            [ws.detected_points[-1]],
            [ws.image_size[-1]],
            fix_aspect=camera_opts.fix_aspect,
            has_skew=camera_opts.allow_skew,
            max_images=camera_opts.limit_intrinsic,
            isFisheye=camera_opts.isFisheye,
        )
        ws.set_calibration(calib.cameras)

    ws.initialise_poses(
        motion_model=get_motion_model(camera_opts.motion_model),
        camera_poses=calib.camera_poses if calib is not None else None,
        camera_index_pairs=camera_index_pairs[:len(ws.detected_points) -1]
    )
    return ws


def optimize(ws: Workspace, opt: OptimizerOpts = OptimizerOpts(), name="calibration"):

    ws.calibrate(
        name=name,
        loss=opt.loss,
        boards=opt.adjust_board,
        cameras=not opt.fix_intrinsic,
        camera_poses=not opt.fix_camera_poses,
        board_poses=not opt.fix_board_poses,
        motion=not opt.fix_motion,
        auto_scale=opt.auto_scale,
        outlier_threshold=opt.outlier_threshold,
        quantile=opt.outlier_quantile,
    )

    return ws
