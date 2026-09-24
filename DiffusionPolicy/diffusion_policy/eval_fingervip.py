from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(ROOT_DIR / "DiffusionPolicy"))

import hydra
import dill
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.process_data import get_image_transform
import meshcat.geometry as g
import meshcat.transformations as tf
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

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

class HWDiffusionPolicyEval:
    """Prepare checkpoint-defined observations and run the policy on CUDA."""

    def __init__(self,
                 ckpt_path: str | Path, # diffusion policy checkpoint path
                 handbase_transform_mask: np.ndarray = np.ones(5, dtype=int),
                 in_res: tuple = (640, 480),
                 out_res: tuple = (224, 224),
                 urdf_file_name: str = "ur5e_with_rapid_hand_right_fingercamera",
                 random_disabling: bool = False,
                 random_disabling_prob: float = 0.1,
                 random_latency: bool = False,
                 random_latency_prob: float = 0.2,
                 random_latency_range: Tuple[float, float] = (0, 1.0),
                 store_attention: bool = False,  # store attention maps
                 render: bool = False,
                 use_velocity: bool = False,
                 use_current: bool = True,
                 use_third_view: bool = True,
                ) -> None:
        
        # Retain caller options for compatibility; they do not currently alter preprocessing.
        self.random_disabling = random_disabling
        self.random_disabling_prob = random_disabling_prob
        self.random_latency= random_latency
        self.random_latency_prob = random_latency_prob
        self.random_latency_range = random_latency_range

        self.use_current = use_current
        self.use_velocity = use_velocity
        self.use_third_view = use_third_view

        # Observation history, initialized by the first prediction frame.
        self.last_obs_dict_np = None
        self.render = render

        # Accept a checkpoint file or a run directory containing checkpoints/latest.ckpt.
        ckpt_path = os.fspath(ckpt_path)
        if not ckpt_path.endswith('.ckpt'):
            ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
        with open(ckpt_path, 'rb') as checkpoint_file:
            payload = torch.load(checkpoint_file, map_location='cpu', pickle_module=dill)
        cfg = payload['cfg']
        print("model_name: ", cfg.policy.obs_encoder.model_name)
        print("dataset_path:", cfg.task.dataset.dataset_path)

        self.obs_shape_meta = cfg.task.shape_meta['obs']
        rgb_keys = list()
        lowdim_keys = list()
        for key, attr in self.obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'rgb':
                rgb_keys.append(key)
            elif obs_type == 'low_dim':
                lowdim_keys.append(key)

        self.camera_ids = []
        camera_key = re.compile(r'^camera(\d+)_rgb$')
        for item in rgb_keys:
            m = camera_key.match(item)
            if m:
                camera_id = int(m.group(1))
                if not 1 <= camera_id <= 5:
                    raise ValueError("FingerViP camera IDs must be 1 through 5.")
                self.camera_ids.append(camera_id)
        if len(self.camera_ids) > 0:
            self.camera_ids = sorted(self.camera_ids)

        # Restore the workspace and select the checkpoint's policy or EMA policy.
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model
        self.policy.num_inference_steps = 16

        self.img_obs_horizon = cfg.task.img_obs_horizon
        self.low_dim_obs_horizon = cfg.task.low_dim_obs_horizon  # same as img_obs_horizon
        
        self.device = torch.device('cuda')
        self.policy.eval().to(self.device)

        print("policy inference")
        self.policy.reset()
        if store_attention:
            dec_attn_weights = []
            self.hooks = [
                self.policy.model.decoder.layers[-1].multihead_attn.register_forward_hook(
                    lambda self, input, output: dec_attn_weights.append(output[1])
                )
            ]

        self._init_policy_training(
            in_res=in_res,
            out_res=out_res,
            urdf_file=f"{urdf_file_name}.urdf",
            handbase_transform_mask=handbase_transform_mask
        ) # For policy inference

    def _init_policy_training(self, 
                              in_res=(640, 480), 
                              out_res=(224, 224), 
                              urdf_file="ur5e_with_rapid_hand_right_fingercamera.urdf",
                              handbase_transform_mask=np.ones(5, dtype=int)
                              ):
        self.in_res = tuple(in_res)
        self.resize_tf = get_image_transform(in_res, out_res, crop_ratio_h=1.0, crop_ratio_w=1.0)

        urdf_path = str(Path(__file__).parent.parent.resolve() / f"dexterous_hand/urdf/{urdf_file}")
        mesh_dir = str(Path(__file__).parent.parent.resolve() / "dexterous_hand/assets")
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()

        transform_mask = np.asarray(handbase_transform_mask)
        if transform_mask.shape != (5,) or not np.isin(transform_mask, [0, 1]).all():
            raise ValueError("handbase_transform_mask must contain five binary values in fingertip order.")
        self.handbase_camera_index = np.where(transform_mask == 1)[0]

        self.tip_frame_names = ["thtip_camera",
                           "intip_camera", 
                           "mitip_camera", 
                           "ritip_camera", 
                           "pitip_camera"]
        
        self.tip_camera_names = ["TD_camera",
                                 "ID_camera", 
                                 "MD_camera", 
                                 "RD_camera", 
                                 "LD_camera"]  # fingertip cameras

        self.reorder_idx = [
            1, 0, 2, 3,     # Index finger: IF, IS, IP, ID
            13, 12, 14, 15, # Little finger: LF, LS, LP, LD
            5, 4, 6, 7,     # Middle finger: MF, MS, MP, MD
            9, 8, 10, 11,   # Ring finger: RF, RS, RP, RD
            17, 16, 18, 19  # Thumb: TF, TS, TP, TD
        ]

        if self.render:
            self.geom_model_vis = pin.buildGeomFromUrdf(self.pin_model, urdf_path, pin.GeometryType.VISUAL, None, [mesh_dir])
            self.geom_model_col = pin.buildGeomFromUrdf(self.pin_model, urdf_path, pin.GeometryType.COLLISION, None, [mesh_dir])
            self.geom_data_vis = pin.GeometryData(self.geom_model_vis)
            self.geom_data_col = pin.GeometryData(self.geom_model_col)

            # === Initialize MeshCat visualizer ===
            self.viz = MeshcatVisualizer(self.pin_model, self.geom_model_col, self.geom_model_vis)
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel()
        

    def real_time_prepare_data_for_policy(self,
                                          arm_qpos = np.zeros(6),
                                          hand_qpos = np.zeros(20),
                                          arm_vel = np.zeros(6),
                                          hand_vel = np.zeros(20),
                                          arm_cur = np.zeros(6),
                                          hand_cur = np.zeros(20),
                                          rgbs = np.zeros((5, 640, 480, 3), dtype=np.uint8),
                                          tv_rgb = np.zeros((1, 640, 480, 3), dtype=np.uint8)
                                          ):
        if 'joint_pos' in self.obs_shape_meta:
            frame_data = {
                'joint_pos': np.concatenate([arm_qpos, hand_qpos], axis=0), # (26,)
            }

        if 'hand_vel' in self.obs_shape_meta:
            frame_data['hand_vel'] = np.asarray(hand_vel) # (20,)
        if 'arm_vel' in self.obs_shape_meta:
            frame_data['arm_vel'] = np.asarray(arm_vel) # (6,)

        if 'arm_cur' in self.obs_shape_meta:
            frame_data['arm_cur'] = np.asarray(arm_cur) # (6,)

        hand_cur = hand_cur[self.reorder_idx] * 2.69e-6 # A
        hand_qpos_reorder = hand_qpos[self.reorder_idx]
        qpos = np.concatenate([arm_qpos, hand_qpos_reorder], axis=0) # (26,)
        pin.forwardKinematics(self.pin_model, self.pin_data, qpos)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
                
        cam_pos = [] # (N, 3)
        cam_ori = [] # (N, 6)
        finger_cur = [] # (20, )
        finger_camera_order = [4, 0, 2, 3, 1] # thumb, index, middle, ring, little
        for idx, (tip_name, tip_camera_name) in enumerate(zip(self.tip_frame_names, self.tip_camera_names)):
            fid = self.pin_model.getFrameId(tip_name)
            T = self.pin_data.oMf[fid]
                    
            if idx in self.handbase_camera_index: # apply hand-base transform, pos and orientation on BASE frame
                fid_base = self.pin_model.getFrameId("BASE")
                T_world_base = self.pin_data.oMf[fid_base]
                T = T_world_base.inverse() * T
            cam_idx = finger_camera_order[idx]
            finger_cur.append(hand_cur[cam_idx * 4 : (cam_idx + 1) * 4].copy()[None,:])
            cam_pos.append(T.translation.copy())
            cam_ori.append(T.rotation.copy()[..., :2].reshape(1,6))

        cam_pos = np.concatenate(cam_pos, axis=0) # (N*3,)
        cam_ori = np.concatenate(cam_ori, axis=0) # (N, 6)
        finger_cur = np.concatenate(finger_cur, axis=0) 
        if 'camera_pos' in self.obs_shape_meta:
            if len(self.camera_ids) > 0:
                frame_data['camera_pos'] = cam_pos.reshape(-1, self.obs_shape_meta['camera_pos']['shape'][1])[[camera_id - 1 for camera_id in self.camera_ids], :] # (N, 3)
            else:
                frame_data['camera_pos'] = cam_pos
        if 'camera_ori' in self.obs_shape_meta:
            if len(self.camera_ids) > 0:
                frame_data['camera_ori'] = cam_ori.reshape(-1, self.obs_shape_meta['camera_ori']['shape'][1])[[camera_id - 1 for camera_id in self.camera_ids], :] # (N, 6)
            else:
                frame_data['camera_ori'] = cam_ori # (N, 6)
        
        if 'finger_cur' in self.obs_shape_meta:
            finger_camera_indices = [camera_id - 1 for camera_id in self.camera_ids]
            frame_data['finger_cur'] = finger_cur.reshape(-1, self.obs_shape_meta['finger_cur']['shape'][1])[finger_camera_indices, :] # (N_fingers, 4)

        if self.render:
            pin.updateGeometryPlacements(self.pin_model, self.pin_data, self.geom_model_vis, self.geom_data_vis)
            self.viz.display(qpos)

            # visualize frames
            tip_names = ["thtip_camera", "intip_camera", "mitip_camera", "ritip_camera", "pitip_camera"]
            model_names = self.pin_model.names # Joint names.
            all_frames = [f.name for f in self.pin_model.frames]  # Frame names.
            for name in all_frames:
                fid = self.pin_model.getFrameId(name)
                if name in tip_names:
                    axis_len = 0.05
                else:
                    axis_len = 0.03
                draw_frame(self.viz.viewer, f"/frames/{name}", self.pin_data.oMf[fid], length=axis_len)
        
        if 'tv_rgb' in self.obs_shape_meta:
            frame_data['tv_rgb'] = self.resize_tf(tv_rgb[0]).transpose(2, 0, 1).astype(np.float32) / 255.0

        if len(self.camera_ids) > 0:
            for i in self.camera_ids:
                frame_data[f'camera{i}_rgb'] = self.resize_tf(rgbs[i - 1]).transpose(2, 0, 1).astype(np.float32) / 255.0
        
        return frame_data


    def predict_actions(self, real_arm, real_hand, frame_idx):

        # Camera IDs 1-5 correspond to fingertip array rows 0-4.
        current_rgb = real_hand.get("fingertip_rgb")
        expected_shape = (5, *self.in_res, 3)
        if not isinstance(current_rgb, np.ndarray) or current_rgb.shape != expected_shape:
            raise ValueError(f"fingertip_rgb must be a NumPy array with shape {expected_shape}.")
        if current_rgb.dtype != np.uint8:
            raise TypeError("fingertip_rgb must have dtype uint8.")
        thirdview_rgb = real_hand.get("third_view_rgb")
        if 'tv_rgb' in self.obs_shape_meta:
            expected_shape = (1, *self.in_res, 3)
            if not isinstance(thirdview_rgb, np.ndarray) or thirdview_rgb.shape != expected_shape:
                raise ValueError(f"third_view_rgb is required with shape {expected_shape}.")
            if thirdview_rgb.dtype != np.uint8:
                raise TypeError("third_view_rgb must have dtype uint8.")

        arm_qpos = real_arm["joint_positions"]
        hand_qpos = real_hand["joint_positions"]
        arm_vel = real_arm["joint_velocities"]
        hand_vel = real_hand["joint_velocities"]
        arm_qcur = real_arm["joint_current"]
        hand_qcur = real_hand["joint_current"]

        # for policy inference
        obs_dict_np = self.real_time_prepare_data_for_policy(
            arm_qpos=arm_qpos, # (6,)
            hand_qpos=hand_qpos, # (20,)
            arm_vel=arm_vel, # (6,)
            hand_vel=hand_vel, # (20,)
            arm_cur=arm_qcur,
            hand_cur=hand_qcur,
            rgbs=current_rgb, #   (num_cam, H, W, 3)
            tv_rgb=thirdview_rgb, 
        ) # {..., 'camera1_rgb': (H, W, 3), ..., 'joint_pos': (26, ), 'joint_vel': (26, 1), 'camera_pos': (N, 3), 'camera_ori': (N, 6)}
        
        pred_actions = None
        if frame_idx == 1:
            self.last_obs_dict_np = {k: np.repeat(v[None], self.img_obs_horizon, axis=0) for k, v in obs_dict_np.items()} # (horizon, ...)
        else:
            if self.last_obs_dict_np is None:
                print("[robot] last_obs_dict_np is None!")

        try:
            with torch.no_grad():
                self.last_obs_dict_np= {k: np.concatenate([self.last_obs_dict_np[k][1:], obs_dict_np[k][None]], axis=0) for k in obs_dict_np.keys()}
                obs_dict = dict_apply(self.last_obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
                pred_actions = self.policy.predict_action(obs_dict)['action_pred'][0].detach().cpu().numpy()
        except KeyboardInterrupt:
            print("Interrupted")

        return pred_actions
    

def main() -> None:
    args = parse_args()
    checkpoint_path = str(ROOT_DIR / args.checkpoint_path)
    robot_policy = HWDiffusionPolicyEval(
        ckpt_path=checkpoint_path,
        handbase_transform_mask=np.array(args.handbase_transform_mask),
        in_res=(640, 480),
        out_res=(224, 224),
        urdf_file_name=args.urdf_file_name,
        use_current=True,
        use_third_view=True)
    
    arm_dof = 6 
    hand_dof = 20 
    H = 480
    W = 640
    num_fingertip = 5
    real_arm = dict(
        joint_positions  = np.zeros(arm_dof),
        joint_velocities = np.ones(arm_dof) * 0.1,
        joint_current    = np.ones(arm_dof) * 0.5,
    )

    real_hand = dict(
        joint_positions  = np.zeros(hand_dof),
        joint_velocities = np.ones(hand_dof) * 0.1,
        joint_current    = np.ones(hand_dof) * 0.3,
        third_view_rgb   = np.full((1, W, H, 3), 128, dtype=np.uint8),  # gray image
        fingertip_rgb    = np.full((num_fingertip, W, H, 3), 64, dtype=np.uint8)
    )
    predict_actions = robot_policy.predict_actions(real_arm, real_hand, frame_idx=1)

if __name__ == "__main__":  # pragma: no cover
    def parse_mask(s):
        s = s.replace("[", "").replace("]", "")
        return np.array([int(x) for x in s.split(",")])

    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Diffusion Policy Evaluation"
        )
        parser.add_argument(
            "checkpoint_path",
            type=str,
            help="Path to diffusion policy checkpoint or folder containing checkpoints",
        )
        parser.add_argument(
            "handbase_transform_mask",
            type=parse_mask,
            nargs='?',
            default=[1, 1, 1, 1, 1]
        )

        parser.add_argument(
            "urdf_file_name",
            type=str,
            nargs='?',    
            default="ur5e_with_rapid_hand_right_fingercamera",
        )
        return parser.parse_args()
    
    main()
