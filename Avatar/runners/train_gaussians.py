import os
import statistics
from collections import defaultdict
from PIL import Image
import json
import cv2
import numpy as np
import smplx
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, quaternion_apply
from smplx.utils import Struct
from torch import nn
from torch.optim import Adam, SGD
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm
import time
from main import TrainingStage
from utils.data import dict2device, pass_smplx_dict, rotate_verts
from utils.gaussian_rasterizer import MyGaussianRasterizer
from utils.general import instantiate_from_config, q_normalize, s_act, o_act, s_inv_act
from utils.lr_scheduler import ExponentialLRxyz
from utils.optimizer_utils import cat_tensors_to_optimizer, prune_optimizer
from utils.rasterizer import NVDiffrast
import trimesh
import cv2
import numpy as np
from pytorch3d.ops import knn_points
from plyfile import PlyData, PlyElement

class TextureAndGaussianTrainer(nn.Module):
    BETAS_SHAPE = 10
    # for infant it is 20, original it is 10
    def __init__(
            self,
            render_size,
            texture_size,
            criteria_config,
            smplx_path,
            gender="male",
            lr_texture=0.001,
            color_lr=0.0025,
            position_lr=0.00016,
            final_position_lr=0.0000016,
            opacity_lr=0.05,
            scaling_lr=0.005,
            rotation_lr=0.001,
            pose_lr=0.001,
            pose_hf_scale=10,
            tto_pose_lr=3e-3,
            tto_decay_steps=70,

            min_s_value=0.0,
            max_s_value=0.1,

            gaussians_optimize_steps=8000,
            texture_optimize_steps=2500,
            opacity_optimize_steps=5000,
            tto_optimize_steps=100,

            densify_start=0,
            densify_step=100,
            densify_stop=6000,

            prune_start=0,
            prune_step=301,
            prune_stop=100000,

            use_pca_for_hands=False,
            eval_frequency=1000,
    ):
        super().__init__()

        # Store passed parameters
        self._texture_size = texture_size
        self._lr_texture = lr_texture
        self._color_lr = color_lr
        self._position_lr = position_lr
        self._final_position_lr = final_position_lr
        self._opacity_lr = opacity_lr
        self._scaling_lr = scaling_lr
        self._rotation_lr = rotation_lr
        self._pose_lr = pose_lr
        self._pose_hf_scale = pose_hf_scale
        self._tto_pose_lr = tto_pose_lr
        self._tto_decay_steps = tto_decay_steps

        self._render_size = render_size
        self._gaussians_optimize_steps = gaussians_optimize_steps
        self._texture_optimize_steps = texture_optimize_steps
        self._opacity_optimize_steps = opacity_optimize_steps
        self._tto_optimize_steps = tto_optimize_steps

        self._min_s_value = min_s_value
        self._max_s_value = max_s_value

        self._densify_start = densify_start
        self._densify_step = densify_step
        self._densify_stop = densify_stop

        self._prune_start = prune_start
        self._prune_step = prune_step
        self._prune_stop = prune_stop

        self._use_pca_for_hands = use_pca_for_hands
        self._eval_frequency = eval_frequency

        # Constants I'm to lazy to move to the config
        self._logging_frequency = 10
        self._grad_threshold = 0.00005
        self._scale_th = 0.005
        self._min_opacity = 0.3

        # Make trainable variables
        betas = torch.zeros((1, self.BETAS_SHAPE), requires_grad=True)
        self._betas = torch.nn.Parameter(betas, requires_grad=True)
        self._initialize_texture()

        self._body_pose_dict = torch.nn.ParameterDict()
        self._body_pose_dict_hf = torch.nn.ParameterDict()

        # Create rasterizers
        self._rasterizer = NVDiffrast(render_size)
        self._gaussian_rasterizer = MyGaussianRasterizer(
            render_size=render_size
        )

        # Load parametric models and loses
        self._load_smplx_model(smplx_path, gender)
        self._setup_criteria(criteria_config)
        self.change_parameters_shape(len(self._faces))

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

        self._callbacks = []
        self.logger = None
        self._loaded_training_stage = None



    @property
    def device(self):
        return "cuda"

    def _initialize_texture(self):
        #self._trainable_texture = torch.zeros((3, self._texture_size, self._texture_size))
        prior_texture = cv2.imread("./metadata/init_texture.jpg")[..., ::-1]
        prior_texture = prior_texture[::-1]
        prior_texture = cv2.resize(prior_texture, (self._texture_size, self._texture_size))
        prior_texture = np.transpose(prior_texture, (2, 0, 1)) / 255
        self._trainable_texture = torch.FloatTensor(prior_texture)
        self._trainable_texture = torch.nn.Parameter(self._trainable_texture, requires_grad=True)

    def _setup_criteria(self, criteria_config):
        self._criteria = {}
        for name, config in criteria_config.items():
            criterion = instantiate_from_config(config)
            self._criteria[name] = criterion

    def _load_smplx_model(self, smplx_path, gender):
        model_params = dict(model_path=smplx_path,
                            model_type='smplx',
                            use_pca=self._use_pca_for_hands,
                            use_hands=True,
                            use_face=True,
                            num_pca_comps=12,
                            use_face_contour=False,
                            create_global_orient=False,
                            create_body_pose=False,
                            create_betas=False,
                            create_left_hand_pose=False,
                            create_right_hand_pose=False,
                            create_expression=False,
                            create_jaw_pose=False,
                            create_leye_pose=False,
                            create_reye_pose=False,
                            create_transl=False,
                            flat_hand_mean=False,
                            dtype=torch.float32,
                            )
        self._smplx_model = smplx.create(gender=gender, **model_params)

        # for infant

        #print(smplx_path)
        # model_params = dict(model_path="/mounted/home/dresden/repositories/HAHA/data/smplx/smil_web.pkl",  model_type='smil', data_struct=None,
        #          create_betas=True,
        #          betas=None,
        #          create_global_orient=True,
        #          global_orient=None,
        #          create_body_pose=True,
        #          body_pose=None,
        #          create_transl=True,
        #          transl=None,
        #          dtype=torch.float32,
        #          batch_size=1,
        #          joint_mapper=None, gender='neutral',
        #          vertex_ids=None)

        # self._smplx_model = smplx.create( **model_params)


        # here we may change dimension of beta and expression by self._smplx_model.shape_param_dim =100
        # self._smplx_model.expr_param_dim = 50
        print("model_created")
        self._faces = self._smplx_model.faces_tensor.contiguous().int()

        model_fn = 'SMPLX_{}.{ext}'.format(gender.upper(), ext="npz")
        smplx_path = os.path.join(smplx_path, 'smplx', model_fn)
        model_data = np.load(smplx_path, allow_pickle=True)
        data_struct = Struct(**model_data)

        self._uv = torch.Tensor(data_struct.vt).contiguous()
        self._uv_idx = torch.Tensor(data_struct.ft.astype(np.int32)).contiguous().int()
        self._rasterizer.set_smplx_parameters(self._faces, self._uv, self._uv_idx)

    def change_parameters_shape(self, npoints):
        self._xyz = torch.nn.Parameter(torch.zeros([npoints, 3], device=self.device), requires_grad=True)
        self._color = torch.nn.Parameter(torch.rand([npoints, 3], device=self.device), requires_grad=True)
        self._rotation = torch.nn.Parameter(torch.zeros([npoints, 4], device=self.device), requires_grad=False)
        self._rotation[:, 0] = 1
        self._rotation.requires_grad = True
        self._scaling = torch.nn.Parameter(torch.zeros([npoints, 3], device=self.device) - 1, requires_grad=False)
        self._scaling[:, 1] = -2  # Negative values because of the activation function
        self._scaling.requires_grad = True
        self._opacity = torch.nn.Parameter(torch.ones([npoints, 1], device=self.device) * 6, requires_grad=True)
        # Mapping from faces indexes to Gaussians indexes
        self._gaussian_to_face = torch.nn.Parameter(torch.range(0, npoints - 1, dtype=torch.long, device=self.device),
                                                    requires_grad=False)
        self._xyz_gradient_accum = torch.nn.Parameter(torch.zeros([npoints], device=self.device), requires_grad=False)
        self._xyz_gradient_denom = torch.nn.Parameter(torch.zeros([npoints], device=self.device), requires_grad=False)
        self._max_radii2D = torch.nn.Parameter(torch.zeros([npoints], device=self.device), requires_grad=False)

    def load_checkpoint(self, pretrain_path):   
        checkpoint_data = torch.load(pretrain_path)
        state_dict = checkpoint_data['state_dict']

        for key in list(state_dict.keys()):
            if key.startswith('_body_pose_dict'):
                del state_dict[key]
        #state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lpips.net.scaling_layer")}
        #state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_smplx_model.right_hand_components")}
        #state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_smplx_model.left_hand_components")}
        self.change_parameters_shape(len(state_dict['_xyz']))

        npoints = len(state_dict["_xyz"])
        print("Loaded Gaussians:", npoints)
        self.change_parameters_shape(npoints)
        self.load_state_dict(state_dict)

        # RANDOM COLOR VISUALIZATION
        #self._color = torch.nn.Parameter(torch.rand([len(state_dict['_xyz']), 3], device=self.device),
        #                                 requires_grad=True)

        # FILL MESH TEXTURE WITH WHITE COLOR
        #with torch.no_grad():
        #    self._trainable_texture *= 0
        #    self._trainable_texture += 1.0  # 0.8

        training_stage = pretrain_path.split('/')[-1].split('_')[:-1]
        training_stage = '_'.join(training_stage)
        self._loaded_training_stage = TrainingStage(training_stage)

    def save_checkpoint(self, training_stage, step):
        save_folder = os.path.join(self.logger.log_dir, "checkpoints")
        os.makedirs(save_folder, exist_ok=True)
        save_name = str(training_stage).split('.')[-1] + "_" + str(step) + '.ckpt'
        save_path = os.path.join(save_folder, save_name)
        self.save_path = save_folder
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
        }, save_path)

        # Save state_dict
        state_dict = self.state_dict().copy()
        for key in list(state_dict.keys()):
            if key.startswith('_body_pose_dict'):
                del state_dict[key]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lpips")}
        # print(state_dict['_smplx_model.lbs_weights'].shape)
        # print(state_dict['_smplx_model.pose_mean'].shape)
        # for key in list(state_dict.keys()):
        #     print(key)
        # len(state_dict["_xyz"])
        # state_dict = data
        for key in list(state_dict.keys()):
            if key.startswith('_body_pose_dict'):
                del state_dict[key]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lpips")}
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_xyz_gradient")}
        state_dict['_faces'] = state_dict['_smplx_model.faces_tensor']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_smplx")}
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("_max")}
        # print(state_dict['_smplx_model.lbs_weights'].shape)
        # print(state_dict['_smplx_model.pose_mean'].shape)
        for key,value in list(state_dict.items()):
            print(f'{key}: {value.shape}')
            # print(key)
        # len(state_dict["_xyz"])
        data = state_dict

        def tensor_to_list(obj):
            """Recursively convert all Tensors in a dict to Python lists."""
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [tensor_to_list(x) for x in obj]
            elif isinstance(obj, tuple):
                return tuple(tensor_to_list(x) for x in obj)
            else:
                return obj

        data_converted = tensor_to_list(data)

        with open(save_folder + "/state_dict.json", 'w') as f:
            json.dump(data_converted, f)


        # save ply as original gaussian 
        print('xyz: ',self._xyz.shape)
        print('rotation: ',self._rotation.shape)
        print('opacity: ',self._opacity.shape)
        print('color: ',self._color.shape)
        print('scale:',self._scaling.shape)
        save_path_ply = os.path.join(save_folder, str(training_stage).split('.')[-1] + "_" + str(step) + '.ply')
        self.save_path_ply = save_path_ply

        # with open(save_folder+"/state_dict.json",'w') as f:
        #     json.dump(data,f)

        self.save_ply(path = save_path_ply,
                      xyz = self._xyz,
                      color = self._color,
                      opacity= self._opacity,
                      scaling=  self._scaling,
                      rotation=self._rotation)

    def save_ply(self, path,xyz,color,opacity,scaling,rotation):
        # add on code , check shape
        # initalize the sh feature, but not trained, for the sake of saving ply
        N = xyz.shape[0]  # Number of points
        max_sh_degree = 3 

        # Initialize self._features_dc with shape (N, 1, 3) filled with zeros
        features_dc = color.unsqueeze(1)
        self._features_dc = features_dc.clone()
        # Initialize self._features_rest with shape (N, 3, 24) filled with zeros
        num_sh_coeffs = (max_sh_degree + 1) ** 2 - 1  # 15 for max_sh_degree=4
        features_rest = torch.zeros((N, 3, num_sh_coeffs), dtype=torch.float32, device="cuda")
        self._features_rest = features_rest
        print('features_dc :', features_dc.shape)
        print('features_rest :', features_rest.shape)


        #self.feature_dc, self.feature_rest = self.compute_features_from_rgb(self._color)
      
 
        # original code
        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scale = scaling.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
    def compute_features_from_rgb(self, rgb_values, max_sh_degree=4):
        """
        Compute feature_dc (direct color) and feature_rest (higher-order SH coefficients)
        from RGB values.

        Parameters:
            rgb_values (torch.Tensor): Tensor of shape (N, 3) containing RGB values.
            max_sh_degree (int): Maximum SH degree. Default is 4.

        Returns:
            feature_dc (torch.Tensor): SH direct color coefficients of shape (N, 1, 3).
            feature_rest (torch.Tensor): SH higher-order coefficients of shape (N, M, 3), 
                                        where M = (max_sh_degree + 1)**2 - 1.
        """
        # Number of SH coefficients
        num_sh_coeffs = (max_sh_degree + 1) ** 2

        # Initialize features tensor
        N = rgb_values.shape[0]  # Number of points

        # Create feature tensor for SH coefficients
        features = torch.zeros((N, 3, num_sh_coeffs), dtype=torch.float32)

        # Assign direct color (0th band) to feature_dc
        features[:, :, 0] = rgb_values

        # Assign higher-order SH coefficients (feature_rest)
        # These can be initialized as zeros or small random values.
        features[:, :, 1:] = torch.zeros((N, 3, num_sh_coeffs - 1))

        # Split into feature_dc and feature_rest
        feature_dc = features[:, :, 0:1].permute(0, 2, 1).contiguous()  # Shape: (N, 1, 3)
        feature_rest = features[:, :, 1:].permute(0, 2, 1).contiguous()  # Shape: (N, M, 3)

        return feature_dc, feature_rest


    def initialize_optimizable_pose(self, dataset):
        for i in range(dataset._len):
            #print( i,dataset._len )
            data_dict = dataset[i]
            smplx_params = data_dict["smplx_params"]
            pid = data_dict["pid"]

            if i == 0:
                betas = torch.tensor(smplx_params["betas"], requires_grad=True)
                assert betas.shape[1] == self.BETAS_SHAPE
                self._betas = torch.nn.Parameter(betas, requires_grad=True)

            self._body_pose_dict[pid] = torch.nn.ParameterDict()
            self._body_pose_dict_hf[pid] = torch.nn.ParameterDict()

            global_orient = torch.tensor(smplx_params["global_orient"], requires_grad=True, device=self.device)
            self._body_pose_dict[pid]["global_orient"] = torch.nn.Parameter(global_orient, requires_grad=True)
            transl = torch.tensor(smplx_params["transl"], requires_grad=True, device=self.device)
            self._body_pose_dict[pid]["transl"] = torch.nn.Parameter(transl, requires_grad=True)

            body_pose = torch.tensor(smplx_params["body_pose"], requires_grad=True, device=self.device)
            self._body_pose_dict[pid]["body_pose"] = torch.nn.Parameter(body_pose, requires_grad=True)
            self._body_pose_dict[pid]["body_pose_prior"] = torch.nn.Parameter(body_pose, requires_grad=False)

            right_hand_pose = torch.tensor(smplx_params["right_hand_pose"], requires_grad=True, device=self.device)
            self._body_pose_dict_hf[pid]["right_hand_pose"] = torch.nn.Parameter(right_hand_pose, requires_grad=True)
            left_hand_pose = torch.tensor(smplx_params["left_hand_pose"], requires_grad=True, device=self.device)
            self._body_pose_dict_hf[pid]["left_hand_pose"] = torch.nn.Parameter(left_hand_pose, requires_grad=True)

            leye_pose = torch.tensor(smplx_params["leye_pose"], device=self.device)
            self._body_pose_dict_hf[pid]["leye_pose"] = torch.nn.Parameter(torch.zeros_like(leye_pose),
                                                                          requires_grad=False)
            reye_pose = torch.tensor(smplx_params["reye_pose"], device=self.device)
            self._body_pose_dict_hf[pid]["reye_pose"] = torch.nn.Parameter(torch.zeros_like(reye_pose),
                                                                          requires_grad=False)

            jaw_pose = torch.tensor(smplx_params["jaw_pose"], requires_grad=True, device=self.device)
            self._body_pose_dict_hf[pid]["jaw_pose"] = torch.nn.Parameter(jaw_pose, requires_grad=True)
            expression = torch.tensor(smplx_params["expression"], requires_grad=True, device=self.device)
            self._body_pose_dict_hf[pid]["expression"] = torch.nn.Parameter(expression, requires_grad=True)

    def set_callbacks(self, callbacks):
        self._callbacks = callbacks

    def ping_callbacks(self, function_name, outputs=None):
        for callback in self._callbacks:
            method = getattr(callback, function_name)
            if outputs is not None:
                method(self, outputs)
            else:
                method(self)

    def calculate_metrics(self, batch):
        rgb = batch["rasterization"]
        rgb_gt = batch["rgb_image"]

        rgb = torch.clip(rgb, -1, 1)
       
        return {
            "psnr": self.psnr(rgb, rgb_gt).detach().cpu().numpy().item(),
            "ssim": self.ssim(rgb, rgb_gt).detach().cpu().numpy().item(),
            "lpips": self.lpips(rgb, rgb_gt).detach().cpu().numpy().item(),
        }

    def get_trainable_pose_parameters(self, include_betas=False):
        params = []
        for frame_data in self._body_pose_dict.values():
            for v in frame_data.values():
                params.append(v)
        if include_betas:
            params.append(self._betas)

        params_hf = []
        for frame_data in self._body_pose_dict_hf.values():
            for v in frame_data.values():
                params_hf.append(v)

        print("Trainable pose parameters:", len(params))
        print("*" * 10)

        return params, params_hf

    def configure_optimizers_pose_tune(self):
        params, params_hf = self.get_trainable_pose_parameters(include_betas=True)

        optimizer = SGD([
            {'params': params, 'lr': self._tto_pose_lr, 'name': 'smplx_pose_params'},
            {'params': params_hf, 'lr': self._tto_pose_lr / self._pose_hf_scale, 'name': 'smplx_pose_params_hf'},
        ])

        return optimizer

    def configure_optimizers(self):
        params, params_hf = self.get_trainable_pose_parameters(include_betas=True)

        optimizer = Adam([
            {'params': [self._trainable_texture], 'lr': self._lr_texture, 'name': '_trainable_texture'},
            {'params': [self._color], 'lr': self._color_lr, 'name': '_color'},
            {'params': [self._xyz], 'lr': self._position_lr, 'name': '_xyz'},
            {'params': [self._rotation], 'lr': self._rotation_lr, 'name': '_rotation'},
            {'params': [self._scaling], 'lr': self._scaling_lr, 'name': '_scaling'},
            {'params': [self._opacity], 'lr': self._opacity_lr, 'name': '_opacity'},
            {'params': params, 'lr': self._pose_lr, 'name': 'smplx_pose_params'},
            {'params': params_hf, 'lr': self._pose_lr / self._pose_hf_scale, 'name': 'smplx_pose_params_hf'},
        ])

        return optimizer

    def add_new_gaussians(self, optimizer, new_gaussians, selected_indeces):
        # Concat to the existing Gaussians
        optimizable_tensors = cat_tensors_to_optimizer(optimizer, new_gaussians)
        self._xyz = optimizable_tensors["_xyz"]
        self._color = optimizable_tensors["_color"]
        self._rotation = optimizable_tensors["_rotation"]
        self._scaling = optimizable_tensors["_scaling"]
        self._opacity = optimizable_tensors["_opacity"]
        new_gaussian_to_face = self._gaussian_to_face[selected_indeces]
        self._gaussian_to_face = torch.nn.Parameter(
            torch.cat([self._gaussian_to_face, new_gaussian_to_face], dim=0),
            requires_grad=False
        )
        self._xyz_gradient_accum = torch.nn.Parameter(
            torch.zeros(self._xyz.shape[0], device=self._xyz.device),
            requires_grad=False
        )
        self._xyz_gradient_denom = torch.nn.Parameter(
            torch.zeros(self._xyz.shape[0], device=self._xyz.device),
            requires_grad=False
        )
        self._max_radii2D = torch.nn.Parameter(
            torch.zeros(self._xyz.shape[0], device=self._xyz.device),
            requires_grad=False
        )

    def prune_points(self, optimizer, mask):
        valid_points_mask = ~mask
        optimizable_tensors = prune_optimizer(
            optimizer,
            valid_points_mask,
            exclude_names=["smplx_pose_params", "smplx_pose_params_hf", "_trainable_texture"]
        )

        self._xyz = optimizable_tensors["_xyz"]
        self._color = optimizable_tensors["_color"]
        self._rotation = optimizable_tensors["_rotation"]
        self._scaling = optimizable_tensors["_scaling"]
        self._opacity = optimizable_tensors["_opacity"]

        self._gaussian_to_face = torch.nn.Parameter(self._gaussian_to_face[valid_points_mask], requires_grad=False)
        self._xyz_gradient_accum = torch.nn.Parameter(self._xyz_gradient_accum[valid_points_mask], requires_grad=False)
        self._xyz_gradient_denom = torch.nn.Parameter(self._xyz_gradient_denom[valid_points_mask], requires_grad=False)
        self._max_radii2D = torch.nn.Parameter(self._max_radii2D[valid_points_mask], requires_grad=False)

    def prune(self, optimizer):
        prune_mask = (o_act(self._opacity) < self._min_opacity).squeeze()
        self.prune_points(optimizer, prune_mask)

        print("Number of Gaussians:", len(self._xyz))
        print("Prunned:", prune_mask.sum().item())

    def densify(self, optimizer):
        grads = self._xyz_gradient_accum / self._xyz_gradient_denom
        grads[grads.isnan()] = 0.0
        selected_pts_mask = torch.where(grads >= self._grad_threshold, True, False)

        # Clone
        selected_clone_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(s_act(self._scaling, self._min_s_value, self._max_s_value), dim=1).values <= self._scale_th,
        )
        if selected_clone_mask.sum() > 0:
            new_gaussians = {
                "_xyz": self._xyz[selected_clone_mask],
                "_color": self._color[selected_clone_mask],
                "_rotation": self._rotation[selected_clone_mask],
                "_scaling": self._scaling[selected_clone_mask],
                "_opacity": self._opacity[selected_clone_mask],
            }
            self.add_new_gaussians(optimizer, new_gaussians, selected_clone_mask)

        # Split
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((len(self._xyz)), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= self._grad_threshold, True, False)

        selected_split_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(s_act(self._scaling, self._min_s_value, self._max_s_value), dim=1).values > self._scale_th,
        )
        if selected_split_mask.sum() > 0:
            stds = s_act(self._scaling[selected_split_mask][None].repeat(2, 1, 1), self._min_s_value, self._max_s_value)
            stds = stds / 5  # How far from the surface sample new gaussians
            means = torch.zeros((2, stds.size(1), 3), device=self._scaling.device)
            samples = torch.normal(mean=means, std=stds)
            samples[:, :, 1] = torch.abs(samples[:, :, 1])  # Put splitted values outside of the mesh

            act_scaled = s_act(self._scaling[selected_split_mask], self._min_s_value, self._max_s_value)
            with torch.no_grad():
                self._xyz[selected_split_mask] += samples[0]
                self._scaling[selected_split_mask] = s_inv_act(act_scaled / 1.6, self._min_s_value, self._max_s_value)
            new_xyz = samples[1] + self._xyz[selected_split_mask]
            new_scaling = s_inv_act(act_scaled / 1.6, self._min_s_value, self._max_s_value)

            new_gaussians = {
                "_xyz": new_xyz,
                "_color": self._color[selected_split_mask],
                "_rotation": self._rotation[selected_split_mask],
                "_scaling": new_scaling,
                "_opacity": self._opacity[selected_split_mask],
            }
            self.add_new_gaussians(optimizer, new_gaussians, selected_split_mask)
        print("Number of Gaussians:", len(self._xyz))
        print("Clonned:", selected_clone_mask.sum().item(), "Splitted:", selected_split_mask.sum().item())

    def record_xyz_grad_radii(self, viewspace_point_tensor, radii, update_filter):
        # Record the gradient norm, invariant across different poses
        bs = len(viewspace_point_tensor)
        for i in range(bs):
            self._xyz_gradient_accum[update_filter[i]] += torch.norm(
                viewspace_point_tensor.grad[i, update_filter[i], :2], dim=-1, keepdim=False
            )
            self._xyz_gradient_denom[update_filter[i]] += 1
            self._max_radii2D[update_filter[i]] = torch.max(
                self._max_radii2D[update_filter[i]], radii[i, update_filter[i]]
            )
        return

    def calc_faces_transform(self, vertices):
        faces = self._faces.long()
        T = torch.mean(vertices[faces], dim=1)

        sampled = vertices[faces]
        vec1 = sampled[:, 2] - sampled[:, 1]
        vec2 = sampled[:, 0] - sampled[:, 1]
        vec3 = sampled[:, 0] - sampled[:, 2]
        cross = torch.cross(vec1, vec2, dim=-1)
        norm = torch.nn.functional.normalize(cross, eps=1e-6, dim=-1)
        vec1 = torch.nn.functional.normalize(vec1, eps=1e-6, dim=-1)
        prod = torch.cross(vec1, norm, dim=-1)
        prod = torch.nn.functional.normalize(prod, eps=1e-6, dim=-1)
        rotmat = torch.permute(torch.stack([vec1, norm, prod]), (1, 0, 2))
        rotmat = torch.transpose(rotmat, 1, 2)
        R = matrix_to_quaternion(rotmat)

        MAX_SCALE = 0.05
        area = torch.norm(cross, p=2, dim=-1, keepdim=True)
        vec3_length = torch.norm(vec3, p=2, dim=-1, keepdim=True)
        h = area / vec3_length
        k = torch.mean(torch.stack([h, vec3_length]), dim=0) / MAX_SCALE
        return T, R, k

    def predict_smplx_vertices(
            self,
            data_dict,
            optimize_pose=False,
            calc_gaussians=False,
            rotate_angle=None,
    ):
        predicted_parameters = {}
        pids = data_dict["pid"]
        vertices = []
        pose_vectors = []
        for i, pid in enumerate(pids):
            smplx_params = {}
            for k, v in data_dict["smplx_params"].items():
                smplx_params[k] = v[i]

            if optimize_pose:
                smplx_params.update(self._body_pose_dict[pid])
                smplx_params.update(self._body_pose_dict_hf[pid])

            smplx_params["betas"] = self._betas
            # print(smplx_params["transl"].shape)
            # print(smplx_params["global_orient"].shape)
            # print(smplx_params["body_pose"].shape)
            pose_vectors.append(
                torch.cat(
                    [
                        smplx_params["transl"],
                        smplx_params["global_orient"],
                        smplx_params["body_pose"]
                    ], dim=1)[0]
            )

            # Inference SMPL-X
            model_output = pass_smplx_dict(smplx_params, self._smplx_model, "cuda")
            vertices.append(model_output["vertices"])

        predicted_parameters["vertices"] = torch.cat(vertices, dim=0)
        predicted_parameters["camera_matrix"] = data_dict["smplx_params"]["camera_matrix"]
        predicted_parameters["camera_transform"] = data_dict["smplx_params"]["camera_transform"]

        if rotate_angle is not None:
            predicted_parameters["vertices"] = rotate_verts(predicted_parameters["vertices"], rotate_angle)

        # Setup Gaussians parameters
        if calc_gaussians:
            batch_size = len(pids)

            xyz_list = []
            rotatation_list = []
            scaling_list = []
            for frame_id in range(batch_size):
                T, R, k = self.calc_faces_transform(predicted_parameters["vertices"][frame_id])
                T = T[self._gaussian_to_face]
                R = R[self._gaussian_to_face]
                k = k[self._gaussian_to_face]

                xyz_list.append(T + quaternion_apply(R, self._xyz) * k)
                rotation = q_normalize(self._rotation)
                rigid_rotation = quaternion_multiply(R, rotation)
                rotatation_list.append(rigid_rotation)
                scaling_list.append(s_act(self._scaling, self._min_s_value, self._max_s_value) * k)

            predicted_parameters["offset_xyz"] = self._xyz.unsqueeze(0).repeat(batch_size, 1, 1)
            predicted_parameters["gaussians_xyz"] = torch.stack(xyz_list, dim=0)
            predicted_parameters["gaussians_rotations"] = torch.stack(rotatation_list, dim=0)
            predicted_parameters["gaussians_scales"] = torch.stack(scaling_list, dim=0)
            predicted_parameters["gaussians_colors"] = self._color.unsqueeze(0).repeat(batch_size, 1, 1)
            predicted_parameters["gaussians_opacity"] = o_act(self._opacity.unsqueeze(0).repeat(batch_size, 1, 1))
            
            # print(predicted_parameters["gaussians_xyz"].shape)
            # print(predicted_parameters["gaussians_rotations"].shape)
            # print(predicted_parameters["gaussians_opacity"].shape)
            print(predicted_parameters["gaussians_colors"].shape)
            # print(predicted_parameters["gaussians_scales"].shape)


                    #self.save_ply(path = self.save_path_ply,
            #          xyz = predicted_parameters["gaussians_xyz"][0],
            #          color = predicted_parameters["gaussians_colors"][0],
            #          opacity= predicted_parameters["gaussians_opacity"][0],
            #          scaling=  predicted_parameters["gaussians_scales"][0],
            #         rotation=predicted_parameters["gaussians_rotations"][0])
            # HEAT MAP AS COLOR
            """
            xyz = predicted_parameters["gaussians_xyz"]
            dist_sq, nn_ind, _ = knn_points(xyz, xyz, K=6, return_nn=False)
            dist_sq = torch.mean(dist_sq, -1)
            min_val = 0.0000
            max_val = 0.0003
            dist_sq = torch.clip(dist_sq, min_val, max_val)
            dist_sq = (dist_sq - min_val) / (max_val - min_val)
            dist_sq = dist_sq.unsqueeze(-1)
            dist_sq = dist_sq.repeat(1, 1, 3)
            dist_sq = dist_sq.detach().cpu().numpy()
            dist_sq = cv2.applyColorMap((dist_sq[:, :, 0] * 255).astype(np.uint8), cv2.COLORMAP_JET) / 255
            dist_sq = torch.FloatTensor(dist_sq).to(self.device)
            predicted_parameters["gaussians_colors"] = torch.nn.Parameter(dist_sq, requires_grad=True)
            """

        return predicted_parameters

    def _mix_rasterizations(self, data_dict):
        mesh_rasterization = data_dict["mesh_rasterization"]
        gaussian_rasterization = data_dict["gaussian_rasterization"]
        alpha = data_dict["gaussian_alpha"]
        background = data_dict["background"]

        rasterization = gaussian_rasterization * alpha + (1 - alpha) * mesh_rasterization  # gaussian_rasterization
        
        
        #print(alpha.shape)
        #print(alpha)
        #temp_folder = "temp"
        # Loop through each batch and save the mask as an image
        #for batch_idx in range(alpha.size(0)):
            # Extract the mask for the current batch
            #mask = alpha[batch_idx, 0].cpu().detach().numpy()  # Convert to NumPy array
            
            # Convert the mask to uint8 format (0 or 255 for binary image)
            #mask_image = (mask * 255).astype(np.uint8)
            
            # Create a PIL Image and save it
            #image = Image.fromarray(mask_image)
            #image_path = os.path.join(temp_folder, f"mask_batch_{batch_idx+1}.png")
            #image.save(image_path)

            #print(f"Saved mask image: {image_path}")
        
        
        
        merged_mask = torch.clip(alpha + data_dict["mask_uv"], 0, 1)  # alpha
        rasterization = rasterization * merged_mask  + (1 - merged_mask) * background

        return {
            "rasterization": rasterization,
            "merged_mask": merged_mask,
        }

    def _change_lr(self, optimizer, training_stage):
        if training_stage == TrainingStage.INIT_TEXTURE:
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "smplx_pose_params")
            opt_params['lr'] = 0  # Frize smplx pose optimization
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "smplx_pose_params_hf")
            opt_params['lr'] = 0  # Frize smplx pose optimization
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "_opacity")
            opt_params['lr'] = 0
        elif training_stage in [TrainingStage.OPTIMIZE_GAUSSIANS]:
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "smplx_pose_params")
            opt_params['lr'] = self._pose_lr  # Unfrize smplx pose optimization
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "smplx_pose_params_hf")
            opt_params['lr'] = self._pose_lr / self._pose_hf_scale  # Unfrize smplx pose optimization
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "_opacity")
            opt_params['lr'] = 0
        elif training_stage == TrainingStage.FINETUNE_TEXTURE:
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "smplx_pose_params")
            opt_params['lr'] = 0  # Frize smplx pose optimization
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "smplx_pose_params_hf")
            opt_params['lr'] = 0  # Frize smplx pose optimization
        elif training_stage == TrainingStage.OPTIMIZE_OPACITY:
            for item in optimizer.param_groups:
                item["lr"] = 0
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "_opacity")
            opt_params['lr'] = self._opacity_lr  # Start training opacity
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "_color")
            opt_params['lr'] = self._color_lr  # Tune color of Gaussians
        elif training_stage == TrainingStage.FINETUNE_POSE:
            for item in optimizer.param_groups:
                item["lr"] = 0
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "smplx_pose_params")
            opt_params['lr'] = self._tto_pose_lr
            opt_params = next(item for item in optimizer.param_groups if item["name"] == "smplx_pose_params_hf")
            opt_params['lr'] = self._tto_pose_lr / self._pose_hf_scale
    def _render_frame(self, data_dict, training_stage, optimize_pose=True, rotate_angle=None):
        prediced_result = self.predict_smplx_vertices(
            data_dict,
            optimize_pose=optimize_pose,
            calc_gaussians=training_stage in [TrainingStage.OPTIMIZE_GAUSSIANS,
                                              TrainingStage.OPTIMIZE_OPACITY,
                                              TrainingStage.FINETUNE_POSE],
            rotate_angle=rotate_angle,
        )
        
           
         


        data_dict.update(prediced_result)
        data_dict["texture"] = self._trainable_texture
        data_dict.update(self._rasterizer(data_dict))
        print("Do training")
        if training_stage == TrainingStage.INIT_TEXTURE:
            # Train only texture
            data_dict["rasterization"] = data_dict["mesh_rasterization"]
            data_dict["merged_mask"] = data_dict["mask_uv"]
            background = data_dict["background"]
            alpha = data_dict["merged_mask"]
            data_dict["rasterization"] = data_dict["rasterization"] * alpha + (1 - alpha) * background

        elif training_stage in [TrainingStage.OPTIMIZE_GAUSSIANS]:
            # Train only gaussians
            INF_FAR = 1000
            data_dict["mesh_depth"] = torch.ones_like(data_dict["background"]) * INF_FAR
            #for i in data_dict:
            #    print(i, data_dict[i].shape)
            #print(Test)
            data_dict.update(self._gaussian_rasterizer(data_dict))
            data_dict["rasterization"] = data_dict["gaussian_rasterization"]
            data_dict["merged_mask"] = data_dict["gaussian_alpha"]

        elif training_stage == TrainingStage.FINETUNE_TEXTURE:
            # Train only texture
            data_dict["rasterization"] = data_dict["mesh_rasterization"]
            data_dict["merged_mask"] = data_dict["mask_uv"]
            background = data_dict["background"]
            alpha = data_dict["merged_mask"]
            data_dict["rasterization"] = data_dict["rasterization"] * alpha + (1 - alpha) * background

        elif training_stage in [TrainingStage.OPTIMIZE_OPACITY, TrainingStage.FINETUNE_POSE]:
            # Train texture and gaussians
            alpha = data_dict["mask_uv"]
            background = data_dict["background"]
            with torch.no_grad():
                data_dict["background"] = data_dict["mesh_rasterization"] * alpha + (1 - alpha) * background
            data_dict.update(self._gaussian_rasterizer(data_dict))
            data_dict.update(self._mix_rasterizations(data_dict))
    def _render_frame_uv(self, data_dict, training_stage, optimize_pose=True, rotate_angle=None):
        prediced_result = self.predict_smplx_vertices(
            data_dict,
            optimize_pose=optimize_pose,
            calc_gaussians=training_stage in [TrainingStage.OPTIMIZE_GAUSSIANS,
                                              TrainingStage.OPTIMIZE_OPACITY,
                                              TrainingStage.FINETUNE_POSE],
            rotate_angle=rotate_angle,
        )
        # for pid in data_dict['pid']:
        #     print(pid)
           
        img_size = 1024
        predicted_colors = prediced_result["gaussians_colors"]
        B, N, _ = predicted_colors.shape

        template = trimesh.load("metadata/smplx_uv.obj", process=False)
        uvs = template.visual.uv             # shape: (10475, 2)
        faces = template.faces 
        

        uv_image_template = cv2.imread("metadata/smplx_uv.png")   # shape: (H, W, 3)

        h, w = uv_image_template.shape[:2]

            # Convert UVs to pixel coordinates
        uv_pixels = uvs.copy()
        uv_pixels[:, 0] *= (w - 1)
        uv_pixels[:, 1] = (1.0 - uv_pixels[:, 1]) * (h - 1)

            # Build face → list of gaussians
        face_to_gaussians = defaultdict(list)
        for gaussian_idx, face_idx in enumerate(self._gaussian_to_face):
            face_to_gaussians[int(face_idx)].append(gaussian_idx)

        # # Paint UV maps per batch
        # for b in tqdm(range(B), desc="Saving UV maps"):
        #     uv_map = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        #     pid = data_dict['pid'][b]
        #     for face_idx in range(len(faces)):
        #         gaussian_ids = face_to_gaussians.get(face_idx, [])
        #         if not gaussian_ids:
        #             continue

        #         colors = predicted_colors[b, gaussian_ids]  # torch tensor (num_gaussians, 3)
        #         mean_color = torch.clamp(colors.mean(dim=0), 0, 1) * 255
        #         color = tuple(int(c.item()) for c in mean_color)

        #         pts = uv_pixels[faces[face_idx]].astype(np.int32).reshape((-1, 1, 2))
        #         bgr_color = (color[2], color[1], color[0])  # RGB → BGR
        #         cv2.fillConvexPoly(uv_map, pts, bgr_color)

        #     filename = "s{:06}_b{}.png".format(self.global_step, pid)
        #     cv2.imwrite(os.path.join('metadata/render', filename), uv_map)
        


        data_dict.update(prediced_result)
        data_dict["texture"] = self._trainable_texture
        data_dict.update(self._rasterizer(data_dict))
        print("Do training")
        if training_stage == TrainingStage.INIT_TEXTURE:
            # Train only texture
            data_dict["rasterization"] = data_dict["mesh_rasterization"]
            data_dict["merged_mask"] = data_dict["mask_uv"]
            background = data_dict["background"]
            alpha = data_dict["merged_mask"]
            data_dict["rasterization"] = data_dict["rasterization"] * alpha + (1 - alpha) * background

        elif training_stage in [TrainingStage.OPTIMIZE_GAUSSIANS]:
            # Train only gaussians
            INF_FAR = 1000
            data_dict["mesh_depth"] = torch.ones_like(data_dict["background"]) * INF_FAR
            #for i in data_dict:
            #    print(i, data_dict[i].shape)
            #print(Test)
            data_dict.update(self._gaussian_rasterizer(data_dict))
            data_dict["rasterization"] = data_dict["gaussian_rasterization"]
            data_dict["merged_mask"] = data_dict["gaussian_alpha"]

        elif training_stage == TrainingStage.FINETUNE_TEXTURE:
            # Train only texture
            data_dict["rasterization"] = data_dict["mesh_rasterization"]
            data_dict["merged_mask"] = data_dict["mask_uv"]
            background = data_dict["background"]
            alpha = data_dict["merged_mask"]
            data_dict["rasterization"] = data_dict["rasterization"] * alpha + (1 - alpha) * background

        elif training_stage in [TrainingStage.OPTIMIZE_OPACITY, TrainingStage.FINETUNE_POSE]:
            # Train texture and gaussians
            alpha = data_dict["mask_uv"]
            background = data_dict["background"]
            with torch.no_grad():
                data_dict["background"] = data_dict["mesh_rasterization"] * alpha + (1 - alpha) * background
            data_dict.update(self._gaussian_rasterizer(data_dict))
            data_dict.update(self._mix_rasterizations(data_dict))

    def log(self, var_name, var_value):
        if torch.is_tensor(var_value):
            var_value = var_value.detach()
        self.logger.add_scalar(var_name, var_value, self.global_step)

    def fit_training_stage(
            self,
            optimizer,
            train_dataloader,
            val_dataloader,
            training_stage,
            steps,
            scheduler=None,
    ):
        self.start_time = time.time()
        train_iter = iter(train_dataloader)
        print(training_stage)
        pbar = tqdm(range(steps))
        for step in pbar:
            # Iterate train again and again
            try:
                train_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
        
            optimizer.zero_grad()

            loss, outputs = self.training_step(train_batch, training_stage)
            #print(outputs.keys())
            #print(outputs['gaussians_xyz'].detach().cpu().numpy().shape)
            #print(test)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            pbar.set_description(f"{training_stage}. Loss {loss:.4f}")
            self.ping_callbacks("on_train_batch_end", outputs)

            if training_stage in [TrainingStage.OPTIMIZE_GAUSSIANS]:
                if self._densify_start < step < self._densify_stop:
                    self.record_xyz_grad_radii(outputs["viewspace_points"], outputs["radii"],
                                               outputs["visibility_filter"])
                    if (step - self._densify_start + 1) % self._densify_step == 0:
                        self.densify(optimizer)

            if training_stage in [TrainingStage.OPTIMIZE_OPACITY]:
                if self._prune_start < step < self._prune_stop:
                    if (step - self._prune_start + 1) % self._prune_step == 0:
                        self.prune(optimizer)

            if val_dataloader is not None and self.global_step % self._eval_frequency == 0:
                elapsed_time = time.time() - self.start_time

                self.evaluate(val_dataloader, training_stage,elapsed_time)

            self.global_step += 1 

    def fit(self, train_dataloader, val_dataloader=None):
        self.global_step = 0
        #val_dataloader = train_dataloader
        # Train Gaussians
        optimizer = self.configure_optimizers()
        scheduler = ExponentialLRxyz(optimizer, 0, self._gaussians_optimize_steps,
                                     self._position_lr,
                                     self._final_position_lr,
                                     '_xyz')

        training_stage = TrainingStage.OPTIMIZE_GAUSSIANS
        self.save_checkpoint(training_stage, self.global_step)

        self._change_lr(optimizer, training_stage)
        self.fit_training_stage(
            optimizer, train_dataloader, val_dataloader, training_stage, self._gaussians_optimize_steps, scheduler
        )
        if val_dataloader is not None:
            time_sofar = time.time() - self.start_time
            self.evaluate(val_dataloader, training_stage,time_sofar)
        self.save_checkpoint(training_stage, self.global_step)

        # # Train the texture
        # optimizer = self.configure_optimizers()
        # training_stage = TrainingStage.FINETUNE_TEXTURE
        # self._change_lr(optimizer, training_stage)
        # self.fit_training_stage(
        #     optimizer, train_dataloader, val_dataloader, training_stage, self._texture_optimize_steps
        # )
        # if val_dataloader is not None:
        #     self.evaluate(val_dataloader, training_stage)
        # self.save_checkpoint(training_stage, self.global_step)

        # # Train the opacity
        # optimizer = self.configure_optimizers()
        # training_stage = TrainingStage.OPTIMIZE_OPACITY
        # self._change_lr(optimizer, training_stage)
        # self.fit_training_stage(
        #     optimizer, train_dataloader, val_dataloader, training_stage, self._opacity_optimize_steps
        # )
        # if val_dataloader is not None:
        #     self.evaluate(val_dataloader, training_stage)
        # self.save_checkpoint(training_stage, self.global_step)
        
        #self.save_ply(path = self.save_path_ply,
        #              xyz = self._xyz,
        #              color = self._color,
        #              opacity= self._opacity,
        #              scaling=  self._scaling,
        #              rotation= self._rotation)

    def fit_pose(self, test_dataloader):
        optimizer = self.configure_optimizers_pose_tune()
        self.global_step = 0

        batches_num = len(test_dataloader)
        # Test time optimization of the pose
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self._tto_decay_steps * batches_num, gamma=0.5,
        )

        training_stage = TrainingStage.FINETUNE_POSE
        self._change_lr(optimizer, training_stage)
        optimization_steps = self._tto_optimize_steps * batches_num
        self.fit_training_stage(
            optimizer, test_dataloader, None, training_stage, optimization_steps, scheduler
        )

    @torch.no_grad()
    def evaluate(self, val_dataloader, training_stage, time_sofar=0):
        metrics_accumulate = defaultdict(list)
        with torch.no_grad():
            val_iter = iter(val_dataloader)
            for step in tqdm(range(len(val_dataloader))):
                val_batch = next(val_iter)
                outputs, metrics = self.validation_step(val_batch, training_stage)
                for k, v in metrics.items():
                    metrics_accumulate[k].append(v)

                outputs["batch_idx"] = step
                self.ping_callbacks("on_validation_batch_end", outputs)

        output_file = os.path.join(self.save_path, "metrics.txt")  
        # Open the file in write mode and save the dictionary
        with open(output_file, "w") as f:
            for k, v in metrics_accumulate.items():
                criterion_name = k
                value = statistics.mean(v)
                self.log('val_loss/' + criterion_name, value)
                print(criterion_name, ':', value)  # Print to console
                f.write(f"{criterion_name}: {value}\n")  # Write to file

            custom_line = "NGS: " + str(self._xyz.shape)
            f.write(f"\n{custom_line}\n")
            custom_line = "Time: " + str(time_sofar)
            f.write(f"\n{custom_line}\n")
        print(f"Metrics saved to {output_file}")

    @torch.no_grad()
    def test(self, test_dataloader):
        metrics_accumulate = defaultdict(list)
        with torch.no_grad():
            val_iter = iter(test_dataloader)
            for step in tqdm(range(len(test_dataloader))):
                val_batch = next(val_iter)
                assert self._loaded_training_stage is not None, "Test method only works with pretrained checkpoint"
                outputs,metrics  = self.test_step(val_batch, self._loaded_training_stage)
                for k, v in metrics.items():
                    metrics_accumulate[k].append(v)
                outputs["batch_idx"] = step
                self.ping_callbacks("on_test_batch_end", outputs)
            self.ping_callbacks("on_test_end")
        
        for k, v in metrics_accumulate.items():
            
            criterion_name = k
            value = statistics.mean(v)
            #if criterion_name == 'psnr':
            print(criterion_name, ':', value)


    def training_step(self, train_batch, training_stage):
        train_batch = dict2device(train_batch, self.device)
        loss = 0
        self._render_frame(train_batch, training_stage)
        
        # gt_image = train_batch["rgb_image"]
        # render_image = train_batch["rasterization"]
        # predict_mask = (render_image.abs().sum(dim=1, keepdim=True) > 0).float()
        # mask_image = train_batch["mask_image"]
        # criterion = nn.BCELoss()


        # def iou_loss(pred_mask, gt_mask, eps=1e-6):
        #     # Both should be float tensors in [0, 1], shape: [B, 1, H, W]
        #     intersection = (pred_mask * gt_mask).sum(dim=(1, 2, 3))
        #     union = (pred_mask + gt_mask - pred_mask * gt_mask).sum(dim=(1, 2, 3))
        #     iou = (intersection + eps) / (union + eps)
        #     return 1 - iou.mean()  # IoU Loss

        # L_sil = criterion(predict_mask, mask_image)
        for criterion_name, criterion in self._criteria.items():
            local_loss = criterion(train_batch, training_stage=training_stage)
            loss += local_loss

            if self.global_step % self._logging_frequency == 0:
                self.log('train_loss/' + criterion_name, local_loss)

        if self.global_step % self._logging_frequency == 0:
            self.log('monitoring_step', self.global_step)
        # loss += 0.1*iou_loss(predict_mask, mask_image)
        return loss, train_batch

    @torch.no_grad()
    def validation_step(self, val_batch, training_stage):
        metrics = {}
        val_batch = dict2device(val_batch, self.device)
        self._render_frame_uv(val_batch, training_stage, optimize_pose=False)

        for criterion_name, criterion in self._criteria.items():
            local_loss = criterion(val_batch, training_stage=training_stage)
            if torch.is_tensor(local_loss):
                local_loss = local_loss.detach().cpu().numpy().item()
            metrics[criterion_name] = local_loss

        metrics.update(self.calculate_metrics(val_batch))

        return val_batch, metrics

    @torch.no_grad()
    def test_step(self, test_batch, training_stage):
        metrics = {}
        test_batch = dict2device(test_batch, self.device)
        self._render_frame(test_batch, training_stage, optimize_pose=False)

        
        for criterion_name, criterion in self._criteria.items():
            local_loss = criterion(test_batch, training_stage=training_stage)
            if torch.is_tensor(local_loss):
                local_loss = local_loss.detach().cpu().numpy().item()
            metrics[criterion_name] = local_loss

        metrics.update(self.calculate_metrics(test_batch))


        return test_batch,metrics
