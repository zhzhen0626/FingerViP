
import sys
import os
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
from tqdm import tqdm
import cv2
import zarr
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.custom_codecs.imagecodecs_numcodecs import register_codecs, JpegXl
import meshcat.geometry as g
import meshcat.transformations as tf
import argparse
import zipfile
import tempfile
import shutil

register_codecs()
def draw_frame(viewer, name, M, length=0.03, radius=0.001):
    """
    Draw a Pinocchio frame in Meshcat (X red, Y green, Z blue).
    Args:
        viewer: viz.viewer
        name: Meshcat path (for example, '/frames/IDIP').
        M: pinocchio.SE3 object.
    """
    viewer[name].delete()
    R = M.rotation
    t = M.translation

    # --- Build the homogeneous transform ---
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # --- X axis (red)
    viewer[name]["x"].set_object(
        g.Cylinder(length, radius), g.MeshLambertMaterial(color=0xff0000)
    )
    Tx = T @ tf.rotation_matrix(-np.pi/2, [0, 0, 1]) @ tf.translation_matrix([0, length/2, 0])
    viewer[name]["x"].set_transform(Tx)

    # --- Y axis (green)
    viewer[name]["y"].set_object(
        g.Cylinder(length, radius), g.MeshLambertMaterial(color=0x00ff00)
    )
    Ty = T @ tf.translation_matrix([0, length/2, 0])
    viewer[name]["y"].set_transform(Ty)

    # --- Z axis (blue)
    viewer[name]["z"].set_object(
        g.Cylinder(length, radius), g.MeshLambertMaterial(color=0x0000ff)
    )
    Tz = T @ tf.rotation_matrix(np.pi/2, [1, 0, 0]) @ tf.translation_matrix([0, length/2, 0])
    viewer[name]["z"].set_transform(Tz)

def project_pose(config_path, pose: np.ndarray) -> np.ndarray:
    """Convert abstracted RAPID pose space to joint angles (rad)."""
    def load_config(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if config_path is None:
        config_path = os.path.join(pathlib.Path(__file__).parent.parent.parent, "Hardware/rapid_hand_control/servo_driver/rapid_full.json")
        
    config = load_config(config_path)
    jointmap = config["init"]["jointmap"]
    init_pos = np.array(config["init"]["init_pos"], dtype=np.float32)

    prev_pos = curr_pos = init_pos.copy()
    
    delta = np.zeros(len(jointmap))  # Angular displacement for each motor.
    for i, mappings in enumerate(jointmap):
        for motor_idx, weight in mappings:
            delta[motor_idx] += pose[i] * weight / len(mappings)
    
    real_pos = delta + init_pos # for hardware 

    return delta, real_pos


def flip(imgs):
    return np.flip(np.flip(imgs, axis=1), axis=2)

def get_image_transform(in_res, out_res, crop_ratio_h: float=1.0, crop_ratio_w: float=1.0, bgr_to_rgb: bool=False, w_slice_start=None, h_slice_start=None):
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio_h)
    cw = round(ih * crop_ratio_w / oh * ow)
    interp_method = cv2.INTER_AREA

    if w_slice_start is None:
        w_slice_start = (iw - cw) // 2
    else:
        w_slice_start = round(iw * w_slice_start)
    w_slice = slice(w_slice_start, w_slice_start + cw)
    if h_slice_start is None:
        h_slice_start = (ih - ch) // 2
    else:
        h_slice_start = round(ih * h_slice_start)
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((iw, ih, 3))
        # crop
        img = img[w_slice, h_slice, c_slice]
        # resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img
    
    return transform


def zip_zarr(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
        for root, dirs, files in os.walk(src_dir):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, src_dir)
                z.write(full_path, arcname=rel_path)

def generate_replay_buffer_from_zarr(
        zarr_dir, output_path, compression_level=99,
        in_res=(640, 480), out_res=(224, 224),
        handbase_transform_mask=np.ones(5, dtype=int),
        urdf_file="ur5e_with_rapid_hand_right_fingercamera.urdf",
        render=False, use_RAM=False):
    """Convert recordings, replacing existing outputs only after conversion succeeds."""
    options = dict(
        compression_level=compression_level, in_res=in_res, out_res=out_res,
        handbase_transform_mask=handbase_transform_mask, urdf_file=urdf_file,
        render=render,
    )
    if use_RAM:
        return _generate_replay_buffer_from_zarr(
            zarr_dir, output_path, use_RAM=True, **options)

    destination = pathlib.Path(output_path).absolute()
    archive = pathlib.Path(str(destination) + '.zip')
    source = pathlib.Path(zarr_dir).resolve()
    resolved_destination = destination.resolve()
    if source == resolved_destination or resolved_destination in source.parents:
        raise ValueError("The output directory must not contain the input recordings.")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Expected a Zarr directory path: {destination}")
    if archive.is_dir():
        raise IsADirectoryError(f"Expected a ZIP file path: {archive}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work_dir = pathlib.Path(tempfile.mkdtemp(prefix='.zarr-conversion-', dir=destination.parent))
    staged = work_dir / 'dataset.zarr'
    try:
        _generate_replay_buffer_from_zarr(
            zarr_dir, str(staged), use_RAM=False, **options)
    except BaseException:
        shutil.rmtree(work_dir)
        raise

    # Keep both previous outputs until the new directory and ZIP have been installed.
    backups = []
    installed = []
    outputs = [(staged, destination), (pathlib.Path(str(staged) + '.zip'), archive)]
    try:
        for index, (_, target) in enumerate(outputs):
            if os.path.lexists(target):
                backup = work_dir / f'previous-{index}'
                os.replace(target, backup)
                backups.append((backup, target))
        for temporary, target in outputs:
            os.replace(temporary, target)
            installed.append((temporary, target))
    except BaseException:
        try:
            for temporary, target in reversed(installed):
                os.replace(target, temporary)
            for backup, target in reversed(backups):
                os.replace(backup, target)
        except BaseException as rollback_error:
            # Leave the temporary directory intact for recovery if restoration fails.
            raise RuntimeError(f"Could not restore previous outputs; recovery files remain in {work_dir}") from rollback_error
        shutil.rmtree(work_dir)
        raise
    shutil.rmtree(work_dir)
    print(f"Saved ReplayBuffer to {destination} and {archive}")


def _generate_replay_buffer_from_zarr(zarr_dir,
                                     output_path, 
                                     compression_level=99, 
                                     in_res=(640, 480), 
                                     out_res=(224, 224),
                                     handbase_transform_mask=np.ones(5, dtype=int),
                                     urdf_file="ur5e_with_rapid_hand_right_fingercamera.urdf",
                                     render=False,
                                     use_RAM=False,
                                     ):
    '''
    file structure:
    raw_zarr_data/<task_name>/
    ├── 2024-07-10_13-33-48.zarr
    │   ├── control
    │   ├── joint_pos
    │   ├── joint_vel
    │   ├── joint_current
    │   ├── tv_rgb
    │   ├── rgb
    │   ├── timestamp
    ├── ...
    '''

    if use_RAM:
        output_path = output_path + '.zip'
    else:
        output_path = output_path
    print(f"Saving ReplayBuffer to {output_path}")
    if use_RAM:
        if os.path.isdir(output_path):
            raise IsADirectoryError(f"Expected a ZIP file path, found a directory: {output_path}")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    else:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Save to disk
    if use_RAM:
        out_replay_buffer = ReplayBuffer.create_empty_zarr(
            storage=zarr.MemoryStore()
        ) # so much memory will be used if not write to disk immediately
    else:
        out_replay_buffer = ReplayBuffer.create_empty_zarr(
            storage=zarr.DirectoryStore(output_path)  
        ) # directly write to disk

    resize_tf = get_image_transform(in_res, out_res, crop_ratio_h=1.0, crop_ratio_w=1.0)
    
    episode_files = os.listdir(zarr_dir)
    episode_files = [file for file in episode_files if file.endswith(".zarr")]
    episode_files.sort()
    num_frames = sum(len(zarr.open_group(os.path.join(zarr_dir, episode_file))['rgb']) for episode_file in episode_files)

    img_compressor = JpegXl(level=compression_level, numthreads=1)
    num_cam = zarr.open_group(os.path.join(zarr_dir, episode_files[0]))['rgb'].shape[1]

    if num_cam != 5:
        raise ValueError("Expected five fingertip views in rgb, ordered thumb/index/middle/ring/little.")
    transform_mask = np.asarray(handbase_transform_mask)
    if transform_mask.shape != (5,) or not np.isin(transform_mask, [0, 1]).all():
        raise ValueError("handbase_transform_mask must contain five binary values in fingertip order.")
    handbase_camera_index = np.where(transform_mask == 1)[0]

    reorder_idx = [
                1, 0, 2, 3,     # Index finger: IF, IS, IP, ID
                13, 12, 14, 15, # Little finger: LF, LS, LP, LD
                5, 4, 6, 7,     # Middle finger: MF, MS, MP, MD
                9, 8, 10, 11,   # Ring finger: RF, RS, RP, RD
                17, 16, 18, 19  # Thumb: TF, TS, TP, TD
            ]
    for i in range(num_cam):
        out_replay_buffer.data.require_dataset(
            name=f'camera{i+1}_rgb',
            shape=(num_frames,) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    out_replay_buffer.data.require_dataset(
                name=f'tv_rgb',
                shape=(num_frames,) + out_res + (3,),
                chunks=(1,) + out_res + (3,),
                compressor=img_compressor,
                dtype=np.uint8
            )    
    


    urdf_path = str(pathlib.Path(__file__).parent.parent.resolve() / f"dexterous_hand/urdf/{urdf_file}")   # Path to URDF file
    mesh_dir = str(pathlib.Path(__file__).parent.parent.resolve() / "dexterous_hand/assets")               
    pin_model = pin.buildModelFromUrdf(urdf_path)
    # Create associated data container
    pin_data = pin_model.createData()

    if render:
        geom_model_vis = pin.buildGeomFromUrdf(pin_model, urdf_path, pin.GeometryType.VISUAL, None, [mesh_dir])
        geom_model_col = pin.buildGeomFromUrdf(pin_model, urdf_path, pin.GeometryType.COLLISION, None, [mesh_dir])
        geom_data_vis = pin.GeometryData(geom_model_vis)
        geom_data_col = pin.GeometryData(geom_model_col)

        # === Initialize MeshCat visualizer ===
        viz = MeshcatVisualizer(pin_model, geom_model_col, geom_model_vis)
        viz.initViewer(open=True)
        viz.loadViewerModel()

    def process_episode(replay_buffer, episode_path, ):
        zarr_data = zarr.open_group(episode_path)
        if zarr_data['rgb'].shape[1] != num_cam:
            raise ValueError(f"Expected five fingertip views in {episode_path}")
        third_view = zarr_data['tv_rgb']
        expected_tv_shape = (zarr_data['rgb'].shape[0], *in_res, 3)
        # Teleop saves the third view without a camera axis: (T, W, H, 3).
        if third_view.shape != expected_tv_shape:
            raise ValueError(
                f"tv_rgb in {episode_path} must have shape {expected_tv_shape} "
                f"but got {third_view.shape}."
            )

        episode_data = {
            'joint_pos': zarr_data['joint_pos'],
            'joint_vel': zarr_data['joint_vel'],
            'action': zarr_data['control']
        }

        # === Print fingertip poses ===
        tip_frame_names = ["thtip_camera", "intip_camera", "mitip_camera", "ritip_camera", "pitip_camera"]

        cam_pos_list = []
        cam_ori_list = []
        finger_cur_list = []

        _, dof = zarr_data['joint_vel'].shape
        assert dof == 26, f"Expect dof=26 (6 arm + 20 hand), got {dof}"

        # Retain current observations for the training dataset.
        joint_cur_all = np.asarray(zarr_data['joint_current'])
        episode_data['arm_cur'] = joint_cur_all[:, :6]

        for i in range(zarr_data['joint_pos'].shape[0]):
            hand_qpos = zarr_data['joint_pos'][i][6:]# skip the base dof
            arm_qpos = zarr_data['joint_pos'][i][:6]
            hand_cur = joint_cur_all[i][6:].copy() * 2.69e-6 # A
            hand_cur = hand_cur[reorder_idx]
            hand_qpos = hand_qpos[reorder_idx]
            qpos = np.concatenate([arm_qpos, hand_qpos], axis=0)
            pin.forwardKinematics(pin_model, pin_data, qpos) # ,qvel
            pin.updateFramePlacements(pin_model, pin_data)
            

            cam_pos = []
            cam_ori = []
            finger_cur = []
            finger_camera_order = [4, 0, 2, 3, 1]
            for idx, tip_name in enumerate(tip_frame_names):
                fid = pin_model.getFrameId(tip_name)
                T = pin_data.oMf[fid]
                
                if idx in handbase_camera_index: # apply hand-base transform, pos and orientation on BASE frame
                    fid_base = pin_model.getFrameId("BASE")
                    T_world_base = pin_data.oMf[fid_base]
                    T = T_world_base.inverse() * T
                cam_idx = finger_camera_order[idx]
                finger_cur.append(hand_cur[cam_idx * 4 : (cam_idx + 1) * 4].copy()[None,:])
                
                cam_pos.append(T.translation.copy())
                cam_ori.append(T.rotation.copy()[..., :2].reshape(1,6)) # only take first two columns to avoid redundancy

            cam_pos_list.append(np.concatenate(cam_pos, axis=0))
            cam_ori_list.append(np.concatenate(cam_ori, axis=0))
            finger_cur_list.append(np.concatenate(finger_cur, axis=0))

            if render:
                pin.updateGeometryPlacements(pin_model, pin_data, geom_model_vis, geom_data_vis)
                
                # visualize rgb images
                rgb_list = []
                for c in range(num_cam):
                    img = resize_tf(zarr_data['rgb'][i, c])
                    rgb_list.append(img)
                tv_img = resize_tf(third_view[i])
                rgb_list.append(tv_img)  # The sixth image is the third-view camera.
                # Tile images for display, with at most six per row.
                grid_cols = 6
                grid_rows = int(np.ceil((num_cam + 1) / grid_cols))
                img_rows = []
                for r in range(grid_rows):
                    imgs_row = rgb_list[r * grid_cols:(r + 1) * grid_cols]
                    # Pad with blank images to match row widths.
                    while len(imgs_row) < grid_cols:
                        imgs_row.append(np.zeros_like(rgb_list[0]))
                    img_rows.append(np.concatenate(imgs_row, axis=1))
                img_grid = np.concatenate(img_rows, axis=0)
                cv2.imshow("All Cameras", img_grid[..., ::-1])  # RGB->BGR for OpenCV
                key = cv2.waitKey(1)
                if key == 27:  # Exit on Escape.
                    print("🚪 Visualization interrupted by user.")
                    cv2.destroyAllWindows()

                viz.display(qpos)

                # visualize frames
                tip_names = ["thtip_camera", "intip_camera", "mitip_camera", "ritip_camera", "pitip_camera"]
                model_names = pin_model.names # Joint names.
                all_frames = [f.name for f in pin_model.frames]  # Frame names.
                for name in all_frames:
                    fid = pin_model.getFrameId(name)
                    if name in tip_names:
                        axis_len = 0.05
                    else:
                        axis_len = 0.03
                    draw_frame(viz.viewer, f"/frames/{name}", pin_data.oMf[fid], length=axis_len)
        
        episode_data['camera_pos'] = np.stack(cam_pos_list, axis=0)
        episode_data['camera_ori'] = np.stack(cam_ori_list, axis=0)
        episode_data['finger_cur'] = np.stack(finger_cur_list, axis=0)
        for i in range(num_cam):
            episode_data[f'camera{i+1}_rgb'] = np.stack([resize_tf(frame) for frame in zarr_data['rgb'][:, i]], axis=0)

        episode_data['tv_rgb'] = np.stack([
            resize_tf(third_view[i]) for i in range(third_view.shape[0])
        ], axis=0)
        
        replay_buffer.add_episode(episode_data)

    with tqdm(total=len(episode_files)) as pbar:
        for episode_file in episode_files:
            process_episode(out_replay_buffer, os.path.join(zarr_dir, episode_file))
            pbar.update()
    
    if use_RAM:
        # Replace the previous ZIP only after the new archive is fully written and closed.
        with tempfile.TemporaryDirectory(dir=os.path.dirname(output_path) or '.') as temp_dir:
            temp_path = os.path.join(temp_dir, 'dataset.zarr.zip')
            with zarr.ZipStore(temp_path, mode='w') as store:
                out_replay_buffer.save_to_store(store)
            os.replace(temp_path, output_path)
    else:
        zip_zarr(output_path, output_path + ".zip")
    print("Done!")

def parse_mask(s):
        s = s.replace("[", "").replace("]", "")
        return np.array([int(x) for x in s.split(",")])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", 
                        type=str, 
                        default="pick_ball"
                        )
    parser.add_argument("--prefix",
                        type=str, 
                        default=""
                        )
    parser.add_argument("--render", 
                        action="store_true"
                        )
    parser.add_argument("--urdf_file", 
                        type=str, 
                        default="ur5e_with_rapid_hand_right_fingercamera.urdf"
                        )
    parser.add_argument("--handbase_transform_mask",
                        type=parse_mask,
                        default=[1, 1, 1, 1, 1]
                        )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    task_name = args.task_name # "pick_and_place_bottle"
    prefix = args.prefix

    raw_data_path = os.path.join(ROOT_DIR, f"diffusion_policy/data/raw_zarr_data/{task_name}")
    processed_data_path = os.path.join(ROOT_DIR, f"diffusion_policy/data/processed_zarr_zip/{task_name}_{prefix}/dataset.zarr")
    generate_replay_buffer_from_zarr(raw_data_path, 
                                     processed_data_path, 
                                     in_res=(640, 480), 
                                     out_res=(224, 224),
                                     handbase_transform_mask=args.handbase_transform_mask,
                                     urdf_file=args.urdf_file,
                                     render=args.render,
                                     use_RAM=True)
