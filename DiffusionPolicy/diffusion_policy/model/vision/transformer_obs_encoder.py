import copy
import re
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class TransformerObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str='vit_base_patch16_clip_224.openai',
            global_pool: str='',
            transforms: list=None,
            n_emb: int=768,
            pretrained: bool=False,
            frozen: bool=False,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            feature_aggregation: str=None,
            downsample_ratio: int=32,
            add_camera_poses: bool=False,
            imagenet_norm: bool=True,   # normalize input with imagenet mean and std
            concat_camera_poses: bool=False,
            add_joint_current: bool=False,
            concat_joint_current: bool=False,
        ):
        """
        RGB inputs have shape (B, T, C, H, W).
        Low-dimensional inputs have shape (B, T, D), or (B, T, N, D)
        for per-camera poses and per-finger currents.
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )
        self.model_name = model_name

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
            
        # handle feature aggregation
        self.feature_aggregation = feature_aggregation
        if model_name.startswith('vit'):
            if self.feature_aggregation is None:
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation != 'cls':
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = 'cls'
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        if imagenet_norm:
            if transforms is None:
                transforms = []
            transforms.append(torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)
        
        feature_size = None
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                # Reuse the backbone, retaining view-specific projections and transforms.
                if share_rgb_model and len(rgb_keys) > 1:
                    if key == 'tv_rgb':
                        proj = nn.Identity()
                        if feature_size != n_emb :
                            proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                        key_projection_map[key] = proj
                        
                        has_color_jitter = any(isinstance(t, torchvision.transforms.ColorJitter) for t in transform)
                        realsense_transform = None
                        if has_color_jitter:
                            colorjitter_tf = copy.deepcopy(transform[0])
                            colorjitter_tf.brightness = (0.85, 1.15)
                            colorjitter_tf.contrast = (0.8, 1.2)
                            colorjitter_tf.saturation = (0.8, 1.2)
                            colorjitter_tf.hue = (-0.03, 0.03)
                            realsense_transform = torch.nn.Sequential(colorjitter_tf, transform[-1])
                        else:
                            realsense_transform = nn.Identity()
                        key_transform_map[key] = realsense_transform
                    if key == 'camera1_rgb':
                        proj = nn.Identity()
                        n_emb_tmp = n_emb // 2 if concat_camera_poses or concat_joint_current else n_emb
                        if feature_size != n_emb_tmp:
                             proj = nn.Linear(in_features=feature_size, out_features=n_emb_tmp)
                        key_projection_map[key] = proj
                        key_transform_map[key] = transform
                    continue
                
                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_model_map[key] = this_model
                
                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    example_feature_map = this_model(example_img)
                    example_features = self.aggregate_feature(example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                proj = nn.Identity()
                n_emb_tmp = n_emb // 2 if concat_camera_poses or concat_joint_current else n_emb
                if feature_size != n_emb_tmp:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb_tmp)
                key_projection_map[key] = proj
                if key == 'tv_rgb':
                    has_color_jitter = any(isinstance(t, torchvision.transforms.ColorJitter) for t in transform)
                    realsense_transform = None
                    if has_color_jitter:
                        colorjitter_tf = copy.deepcopy(transform[0])
                        colorjitter_tf.brightness = (0.85, 1.15)
                        colorjitter_tf.contrast = (0.8, 1.2)
                        colorjitter_tf.saturation = (0.8, 1.2)
                        colorjitter_tf.hue = (-0.03, 0.03)
                        realsense_transform = torch.nn.Sequential(colorjitter_tf, transform[-1])
                    else:
                        realsense_transform = nn.Identity()
                    key_transform_map[key] = realsense_transform
                else:
                    key_transform_map[key] = transform
            elif type == 'low_dim':
                if not add_camera_poses and (key == 'camera_pos' or key == 'camera_ori'):
                    continue
                if not add_joint_current and ('cur' in key):
                    continue
                dim = shape[-1] if key == 'camera_pos' or key == 'camera_ori' or 'cur' in key else np.prod(shape)
                proj = nn.Identity()
                if key == 'camera_pos' or key == 'camera_ori':
                    if concat_camera_poses and concat_joint_current:
                        n_emb_tmp = n_emb // 6
                    elif concat_camera_poses:
                        n_emb_tmp = n_emb // 4
                    else:
                        n_emb_tmp = n_emb // 2
                elif 'cur' in key:
                    if concat_camera_poses and concat_joint_current:
                        n_emb_tmp = n_emb // 6
                    elif concat_joint_current:
                        n_emb_tmp = n_emb // 2
                    else:
                        n_emb_tmp = n_emb
                else:
                    n_emb_tmp = n_emb
                if dim != n_emb_tmp:
                    proj = nn.Linear(in_features=dim, out_features=n_emb_tmp)
                key_projection_map[key] = proj

                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_projection_map = key_projection_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.add_camera_poses = add_camera_poses
        self.concat_camera_poses = concat_camera_poses
        self.add_joint_current = add_joint_current
        self.concat_joint_current = concat_joint_current

        self.camera_ids = []
        camera_key = re.compile(r'^camera(\d+)_rgb$')
        for item in self.rgb_keys:
            m = camera_key.match(item)
            if m:
                self.camera_ids.append(int(m.group(1)))
        if len(self.camera_ids) > 0:
            self.camera_ids = sorted(self.camera_ids)
        
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        # Return: B, N, C
        
        if self.model_name.startswith('vit'):
            # vit uses the CLS token
            if self.feature_aggregation == 'cls':
                return feature[:, [0], :]
            
            # or use all tokens
            assert self.feature_aggregation is None 
            return feature
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1], keepdim=True)
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1, keepdim=True)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1, keepdim=True)
        else:
            assert self.feature_aggregation is None
            return feature
        
    def _transform_image(self, key, image):
        transform = self.key_transform_map[key]
        if self.training or isinstance(transform, nn.Identity):
            return transform(image)

        # Keep preprocessing, but disable training-time color and blur augmentation.
        for operation in transform:
            if isinstance(operation, (
                torchvision.transforms.ColorJitter,
                torchvision.transforms.RandomApply,
                torchvision.transforms.GaussianBlur,
            )):
                continue
            image = operation(image)
        return image

    def forward(self, obs_dict):
        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # Process the third view separately, then batch the fingertip views.
        if self.share_rgb_model:
            tv_emb = None
            rgb_mapping_keys = self.rgb_keys.copy()
            rgb_horizon = obs_dict[self.rgb_keys[0]].shape[1]
            if any(obs_dict[key].shape[1] != rgb_horizon for key in self.rgb_keys):
                raise ValueError("Time-major camera fusion requires matching RGB history lengths.")
            if 'tv_rgb' in rgb_mapping_keys and len(rgb_mapping_keys) > 1:
                tv_img = obs_dict['tv_rgb']
                B, T = tv_img.shape[:2]
                assert B == batch_size
                assert tv_img.shape[2:] == self.key_shape_map['tv_rgb']
                tv_img = tv_img.reshape(B*T, *tv_img.shape[2:])
                tv_img = self._transform_image('tv_rgb', tv_img)
                raw_feature = self.key_model_map[self.rgb_keys[0]](tv_img)
                feature = self.aggregate_feature(raw_feature)
                tv_emb = self.key_projection_map['tv_rgb'](feature).reshape(B, -1, self.n_emb)  # B, T, n_emb
                rgb_mapping_keys.remove('tv_rgb')
            
            # Keep all cameras from one time step together before advancing in time.
            img = torch.stack([obs_dict[key] for key in rgb_mapping_keys], dim=2) # B, T, N, C, H, W
            B, T, N = img.shape[:3]
            assert B == batch_size
            assert img.shape[3:] == self.key_shape_map[rgb_mapping_keys[0]]
            img = img.reshape(B*T*N, *img.shape[3:])
            img = self._transform_image(rgb_mapping_keys[0], img)
            raw_feature = self.key_model_map[self.rgb_keys[0]](img)
            feature = self.aggregate_feature(raw_feature)
            emb = self.key_projection_map[rgb_mapping_keys[0]](feature)
            if self.concat_camera_poses or self.concat_joint_current:
                assert len(emb.shape) == 3 and emb.shape[0] == B * T * N and emb.shape[-1] == self.n_emb // 2
                emb = emb.reshape(B,-1,self.n_emb // 2)  # 384  
            else:             
                assert len(emb.shape) == 3 and emb.shape[0] == B * T * N and emb.shape[-1] == self.n_emb
                emb = emb.reshape(B,-1,self.n_emb) # 768
            camera_pose_emb = None
            joint_cur_emb = None
            if self.add_camera_poses:
                assert obs_dict['camera_pos'].shape[:2] == (B, T)
                assert obs_dict['camera_ori'].shape[:2] == (B, T)
                camera_pos = obs_dict['camera_pos'].reshape(-1, 3)
                camera_ori = obs_dict['camera_ori'].reshape(-1, 6)
                
                camera_pos_emb = self.key_projection_map['camera_pos'](camera_pos)
                camera_ori_emb = self.key_projection_map['camera_ori'](camera_ori)
                if self.concat_camera_poses and self.concat_joint_current: 
                    camera_pos_emb = camera_pos_emb.reshape(B, -1, self.n_emb // 6) # 128
                    camera_ori_emb = camera_ori_emb.reshape(B, -1, self.n_emb // 6) # 128
                elif self.concat_camera_poses: # Concatenate image, position, and orientation features.
                    camera_pos_emb = camera_pos_emb.reshape(B, -1, self.n_emb // 4) # 192
                    camera_ori_emb = camera_ori_emb.reshape(B, -1, self.n_emb // 4) # 192
                else: # add only: emb + [cam_pos, cam_ori]
                    camera_pos_emb = camera_pos_emb.reshape(B, -1, self.n_emb // 2) # 384
                    camera_ori_emb = camera_ori_emb.reshape(B, -1, self.n_emb // 2) # 384
                camera_pose_emb = torch.cat([camera_pos_emb, camera_ori_emb], dim=-1) # 256/384/768
            if self.add_joint_current:
                if 'finger_cur' in obs_dict:
                    assert obs_dict['finger_cur'].shape[:2] == (B, T)
                    hand_joint_cur = obs_dict['finger_cur'].reshape(-1, 4)
                    hand_joint_cur_emb = self.key_projection_map['finger_cur'](hand_joint_cur)
                if self.concat_camera_poses and self.concat_joint_current:
                    if 'finger_cur' in obs_dict:
                        hand_joint_cur_emb = hand_joint_cur_emb.reshape(B, -1, self.n_emb // 6) # 128
                elif self.concat_joint_current:
                    if 'finger_cur' in obs_dict:
                        hand_joint_cur_emb = hand_joint_cur_emb.reshape(B, -1, self.n_emb // 2) # 384
                else:
                    if 'finger_cur' in obs_dict:
                        hand_joint_cur_emb = hand_joint_cur_emb.reshape(B, -1, self.n_emb) # 768
                
                if 'finger_cur' in obs_dict:
                    joint_cur_emb = hand_joint_cur_emb

            # Handle embedding combination based on flags and availability
            if self.concat_camera_poses and self.concat_joint_current:
                if camera_pose_emb is not None and joint_cur_emb is not None:
                    emb = torch.cat([emb, camera_pose_emb, joint_cur_emb], dim=-1)
                elif camera_pose_emb is not None:
                    emb = torch.cat([emb, camera_pose_emb], dim=-1)
                elif joint_cur_emb is not None:
                    emb = torch.cat([emb, joint_cur_emb], dim=-1)
            else:
                if self.concat_camera_poses:
                    if camera_pose_emb is not None:
                        emb = torch.cat([emb, camera_pose_emb], dim=-1)
                    if joint_cur_emb is not None:
                        emb = emb + joint_cur_emb
                elif self.concat_joint_current:
                    if joint_cur_emb is not None:
                        emb = torch.cat([emb, joint_cur_emb], dim=-1)
                    if camera_pose_emb is not None:
                        emb = emb + camera_pose_emb
                else:
                    if self.add_camera_poses:
                        emb = emb + camera_pose_emb # 768 + 768
                    if self.add_joint_current:
                        emb = emb + joint_cur_emb # 768 + 768
            
            if tv_emb is not None:
                # Prepend the third-person view within each time step, after hand-view fusion.
                emb = torch.cat([
                    tv_emb.reshape(B, T, -1, self.n_emb),
                    emb.reshape(B, T, -1, self.n_emb),
                ], dim=2).flatten(1, 2)
            embeddings.append(emb)
        else:
            for key in self.rgb_keys:
                img = obs_dict[key]
                B, T = img.shape[:2]
                assert B == batch_size
                assert img.shape[2:] == self.key_shape_map[key]
                img = img.reshape(B*T, *img.shape[2:])
                img = self._transform_image(key, img)
                raw_feature = self.key_model_map[key](img)
                feature = self.aggregate_feature(raw_feature)
                emb = self.key_projection_map[key](feature)
                assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
                emb = emb.reshape(B,-1,self.n_emb)
                embeddings.append(emb)

        # process lowdim input
        for key in self.low_dim_keys:
            if key == 'camera_pos' or key == 'camera_ori' or 'cur' in key:
                continue
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.reshape(B,T,-1)
            emb = self.key_projection_map[key](data)
            assert emb.shape[-1] == self.n_emb
            embeddings.append(emb)
        
        # Concatenate observation tokens along the sequence dimension.
        result = torch.cat(embeddings, dim=1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 3
        assert example_output.shape[0] == 1

        return example_output.shape
