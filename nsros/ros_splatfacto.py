from dataclasses import dataclass, field
import time
import numpy as np
from typing import Type, List, Dict
from typing import Union
import torch

import viser
import viser.theme
import viser.transforms as vtf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nerfstudio.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
    RGB2SH,
    num_sh_bases,
    random_quat_tensor,
)
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.cameras.cameras import Cameras

import pdb

def plot_axes(ax, matrix, origin, label_prefix, color):
    # Extract rotation part of the matrix
    R = matrix[:3, :3]
    # Define unit vectors for axes
    unit_vectors = np.eye(3)
    # Transform unit vectors
    transformed_vectors = np.dot(R, unit_vectors)
    # Plot axes
    for i, vec in enumerate(transformed_vectors.T):
        ax.quiver(origin[0], origin[1], origin[2], vec[0], vec[1], vec[2], length=0.2, color=color[i], linewidth=2, label=f'{label_prefix}{["X", "Y", "Z"][i]}')


@dataclass
class ROSSplatfactoModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: ROSSplatfactoModel)
    depth_seed_pts: int = 4000
    """ Number of points to use for seeding the model from depth per image. """
    seed_with_depth: bool = True
    """ Whether to seed the model from RGBD images. """


class ROSSplatfactoModel(SplatfactoModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Using GPU:", torch.cuda.get_device_name(0))
        device: Union[torch.device, str] = "cuda:0"
        self.seeded_img_idx = 0
        self.depth_seed_pts = self.config.depth_seed_pts
        self.seed_with_depth = self.config.seed_with_depth
        self.vis_counts = None
        self.inter=0
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
    
    def seed_cb(self, pipeline: Pipeline, optimizers: Optimizers, step: int):
        
        if not pipeline.datamanager.train_image_dataloader.listen_depth:
            print("not listen_depth ")
            return
        ds_latest_idx = pipeline.datamanager.train_image_dataloader.current_idx
        if self.seeded_img_idx < ds_latest_idx:
            start_idx = 0 if self.seeded_img_idx == 0 else self.seeded_img_idx + 1
            seed_image_idxs = range(start_idx, ds_latest_idx + 1)
            pre_gaussian_count = self.means.shape[0]
            for idx in seed_image_idxs:
                image_data = pipeline.datamanager.train_dataset[idx]
                # print(f"Contents of train_dataset at index {idx}:")
                # for key, value in image_data.items():
                #     # For tensor data, print its shape and data type
                #     if isinstance(value, torch.Tensor):
                #         # print(f"{key}: shape = {value.shape}, dtype = {value.dtype}")
                #     else:
                #         # print(f"{key}: {value}")

                # If you also need to inspect the camera object:
                camera = pipeline.datamanager.train_dataset.cameras[idx]
                # print(f"Camera data at index {idx}: {camera}")
                with torch.no_grad():
                    self.seed_from_xyzrgb(camera, image_data ,optimizers)
                self.seeded_img_idx = idx
            post_gaussian_count = self.means.shape[0]
            diff_gaussians = post_gaussian_count - pre_gaussian_count

            if self.xys_grad_norm is not None:
                device = self.xys_grad_norm.device
                self.xys_grad_norm = torch.cat(
                    [self.xys_grad_norm, torch.zeros(diff_gaussians).to(device)]
                )
            if self.max_2Dsize is not None:
                device = self.max_2Dsize.device
                self.max_2Dsize = torch.cat(
                    [self.max_2Dsize, torch.zeros(diff_gaussians).to(device)]
                )
            if self.vis_counts is not None:
                device = self.vis_counts.device
                self.vis_counts = torch.cat(
                    [self.vis_counts, torch.zeros(diff_gaussians).to(device)]
                )



# import numpy as np

    def seed_from_xyzrgb(
        self,
        camera: Cameras,
        # image_data: Dict[str, torch.Tensor],
        xyzrgb_tensor: torch.Tensor,
        optimizers: Optimizers,
        device: Union[torch.device, str] = "cpu",
        
        
    ):
        """
        Initialize gaussians at points in the point cloud from the xyzrgb data.

        Args:
            camera: Cameras object containing camera parameters.
            xyzrgb_tensor: Tensor containing XYZ coordinates and RGB values.
            optimizers: Optimizers for updating gaussians parameters.
        """
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # Define colors for the original and transformed axes
        # Define colors for the original and transformed axes
        original_colors = ['r', 'g', 'b']  # RGB for XYZ axes
        transformed_colors = ['c', 'm', 'y']  # CMY for transformed XYZ axes
        # Plot original Y-up axes (assuming identity matrix for original orientation)
        # Plot original Y-up axes (assuming identity matrix for original orientation)

        # rotation_matrix_z_up = torch.tensor([[1, 0, 0, 0],
        #                                     [0, 0, 1, 0],
        #                                     [0, -1, 0, 0],
        #                                     [0, 0, 0, 1]], dtype=torch.float32, device='cuda:0')
        plot_axes(self.ax,np.eye(4), [0, 0, 0], 'Z-up ', original_colors)
        # if xyzrgb_tensor.device != self.device:
        xyzrgb_tensor = xyzrgb_tensor['xyzrgb'].to(self.device)
        xyzs = xyzrgb_tensor[:, :3]
        rgbs = xyzrgb_tensor[:, 3:]
       

        
        # print("--------------------Got POINTS--------------------")
        # print(xyzs)
        # print(rgbs) 
        # print("--------------------End POINTS--------------------")
        # Transform XYZ coordinates to world coordinates using camera extrinsics
        assert len(camera.shape) == 0
        c2w = camera.camera_to_worlds.to(self.device)  # (3, 4)
        # R = c2w[:3, :3]
        # t = c2w[:3, 3].squeeze()
                # Define the rotation matrix for a 180 degree rotation around the x-axis
        # Rx_pi = torch.tensor([
        #     [1, 0, 0],
        #     [0, np.cos(np.pi), -np.sin(np.pi)],
        #     [0, np.sin(np.pi), np.cos(np.pi)]
        # ], dtype=torch.float).to(self.device) 
        
        T = np.array([
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]
                    ])
        # T2 = np.array([
        #             [0, 1, 0, 0],
        #             [-1, 0, 1, 0],
        #             [0, 0, 1, 0],
        #             [0, 0, 0, 1]
        #             ])
        # T3 = np.array([[1, 0, 0, 0],
        #             [0, 0, 1, 0],
        #             [0, -1, 0, 0],
        #             [0, 0, 0, 1]
        #             ])
        
        T_torch = torch.tensor(T, dtype=torch.float32, device=self.device)
        # # T2_torch = torch.tensor(T2, dtype=torch.float32, device=self.device)
        # # T3_torch = torch.tensor(T3, dtype=torch.float32, device=self.device)
        c2w_homogeneous = torch.cat([c2w, torch.tensor([[0, 0, 0, 1]], device=self.device)], dim=0)
        transformed_c2w = torch.matmul(T_torch, c2w_homogeneous)
        # # transformed_c2w = torch.matmul(T2_torch, transformed_c2w)
        # # transformed_c2w = torch.matmul(T3_torch, transformed_c2w)
       
        R = transformed_c2w[:3, :3]
        t = transformed_c2w[:3, 3].squeeze()
        # R = torch.tensor(R, dtype=torch.float32).to(self.device)
        # xyzs_r = torch.matmul(xyzs, R.T) + t  # Rotate 

        
        
        # points_y_up = np.dot(R.cpu().numpy(), )
        # points_y_up_x_inv=np.dot(points_y_up,reflection_matrix)
        # points_inv_rot=np.dot(points_y_up,rot_matrix)
        # xyzs = xyzs


        # inter = inter+1
        # plot_axes(self.ax, R, t, 'Camera ', transformed_colors)
        # # # Assuming xyzrgb_tensor is your tensor of 3D points
        # xyzs_view = xyzs.cpu().numpy()  # Convert to NumPy array if it's a tensor  
        # xyzs_r_view = (xyzs_r).cpu().numpy()  # Convert to NumPy array if it's a tensor  
        # rgbs_view =rgbs.cpu().numpy()
        # self.ax.scatter(xyzs_r_view[:, 0], xyzs_r_view[:, 1], xyzs_r_view[:, 2], c=rgbs_view, marker='.', s=1)  # Use a small size for a large number of points
        # self.ax.scatter(xyzs_view[:, 0], xyzs_view[:, 1], xyzs_view[:, 2], c=rgbs_view, marker='.', s=1)  # Use a small size for a large number of points
        # max_range = np.array([xyzs[:, 0].max()-xyzs[:, 0].min(), xyzs[:, 1].max()-xyzs[:, 1].min(), xyzs[:, 2].max()-xyzs[:, 2].min()]).max() / 2.0
        # mid_x = (xyzs[:, 0].max()+xyzs[:, 0].min()) * 0.5
        # mid_y = (xyzs[:, 1].max()+xyzs[:, 1].min()) * 0.5
        # mid_z = (xyzs[:, 2].max()+xyzs[:, 2].min()) * 0.5

        # self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
        # self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        #         # Set labels and show plot
        # if self.inter == 1:
        #     # Annotation for +X direction (look-at)
        #     self.ax.text(1, 0, 0, "+X (Look-at)", color='red')

        #     # Annotation for -X direction (back)
        #     self.ax.text(-1, 0, 0, "-X (Back)", color='red')

        #     # Annotation for +Z direction (up)
        #     self.ax.text(0, 0, 1, "+Z (Up)", color='blue')

        #     # Annotation for +Y direction (left - assuming left from the viewer's perspective)
        #     self.ax.text(0, 1, 0, "+Y (Left)", color='green')

        #     self.ax.set_xlabel('X axis')
        #     self.ax.set_ylabel('Y axis')
        #     self.ax.set_zlabel('Z axis')
        #     plt.title('Original Y-up and Transformed Z-up Coordinate Systems')
        #     plt.show()
        #     self.inter = 0
        # else:
        #     self.inter += 1  # Increment the counter
        
        # R_new = R
        # R_new[:,2] = R[:,1]
        # R_new[:,1] = R[:,2]
        # print(R,R_new)
        # t_new = t
        # t[2] = t[1]
        # t[1] = t[2]
        # print(t,t_new)
        # R[:, 1] = 
        # rotation_matrix_z_up = np.array([
        # [1., 0., 0.],  # Map LiDAR's forward (x) to camera's forward (z)
        # [0., 0., 1.], # Map LiDAR's left (y) to camera's right (x)
        # [0., 1., 0.], # Map LiDAR's up (z) to camera's down (y)
        # ])
    
        # # Convert R and t to Z-up
        # R_z_up = np.dot(rotation_matrix_z_up, R.cpu().numpy())  # Rotate R
        # t_z_up = np.dot(rotation_matrix_z_up, t.cpu().numpy())  # Rotate t
        #     # TF = torch.eye(4, device=R.device)
        # R_rotation = np.array([
        # [0., 1., 0.],  # Map LiDAR's forward (x) to camera's forward (z)
        # [0., 0., 1.], # Map LiDAR's left (y) to camera's right (x)
        # [1., 0., 0.], # Map LiDAR's up (z) to camera's down (y)
        # ])
        # R_rotation = np.array([
        # [0, -1., 0.],  # Map LiDAR's forward (x) to camera's forward (z)
        # [0., 0., -1.], # Map LiDAR's left (y) to camera's right (x)
        # [-1., 0., 0.], # Map LiDAR's up (z) to camera's down (y)
        # ])
        # R_rotation = np.array([
        # [1., 0., 0.],  # Map LiDAR's forward (x) to camera's forward (z)
        # [0., 1., 0.], # Map LiDAR's left (y) to camera's right (x)
        # [0., 0., 1.], # Map LiDAR's up (z) to camera's down (y)
        # ])
        y_ccw_90 = np.array([
        [0., 0., -1.],  # Map LiDAR's forward (x) to camera's forward (z)
        [0., 1., 0.], # Map LiDAR's left (y) to camera's right (x)
        [-1., 0., 0.], # Map LiDAR's up (z) to camera's down (y)
        ])
        x_ccw_90 = np.array([
        [1., 0., 0.],  # Map LiDAR's forward (x) to camera's forward (z)
        [0., 0., 1.], # Map LiDAR's left (y) to camera's right (x)
        [0., -1., 0.], # Map LiDAR's up (z) to camera's down (y)
        ])
        # x_ccw_180 = np.array([
        # [1., 0., 0.],  # Map LiDAR's forward (x) to camera's forward (z)
        # [0., -1., 0.], # Map LiDAR's left (y) to camera's right (x)
        # [0., 0., -1.], # Map LiDAR's up (z) to camera's down (y)
        # ])
        # filp= np.array([
        # [1., 0., 0.],  # Map LiDAR's forward (x) to camera's forward (z)
        # [0., 1., 0.], # Map LiDAR's left (y) to camera's right (x)
        # [0., 0., -1.], # Map LiDAR's up (z) to camera's down (y)
        # ])
        # R[1]=-R[1]
        # R[2]=-R[2]
        # # R[]
        U_new, Sigma_new, VT_new = np.linalg.svd(R.cpu().numpy())
        R_prime_new = np.dot(U_new, VT_new)
        R_prime_y_90 = np.dot(R_prime_new,y_ccw_90)
        R_prime_y_90_z_ccw_90 =  np.dot(R_prime_y_90,x_ccw_90)
        # R_prime_y_90_z_ccw_90_x_ccw_180 = np.dot(R_prime_y_90_z_ccw_90,x_ccw_180)
        # print(R_prime_y_90_z_ccw_90.round())
        # R_prime_y_90_z_ccw_90_flip = np.dot(filp,R_prime_y_90_z_ccw_90)
        # R_combined=np.dot(R_rotation, R_prime_new)
        # t_z_up = torch.tensor(t_z_up, dtype=torch.float32).to(self.device)
        R_rotation = torch.tensor(R_prime_y_90_z_ccw_90, dtype=torch.float32).to(self.device)
        # R_rotation = torch.from_numpy(R_rotation).to(self.device)

        # ############### want to rotate xyz to world###########################3
        # TF= TF[:, [1, 2, 0, 3]]
        # TF[:, [0, 2]] *= -1
        # [[ 0. -1.  0.]
        # [ 0.  0.  1.]
        # [-1.  0.  0.]]
        # ones = torch.ones((xyzs.shape[0], 1), device=xyzs.device, dtype=xy    zs.dtype)
        # xyzs_homogeneous = torch.cat((xyzs, ones), dim=1)  # Now (N, 4)# Apply transformation
        # transformed_xyzs = torch.matmul(xyzs_homogeneous, TF.T)  # Transpose T if needed
        # xyzs = transformed_xyzs[:, :3]
#
        #############################################################################
        #
        # # Transform y and z: x stays the same, y and z are multiplied by -1
        # print("--------------------R_prime_y_90_z_ccw_90 POINTS--------------------")
        # print(R_prime_y_90_z_ccw_90.round())
        # print("--------------------end POINTS--------------------")
        # xyzs[:, 0] = -xyzs[:, 0]  # Negate x
        # xyzs[:, 1] = -xyzs[:, 1]  # Negate ys

        xyzs[:, 1] = -xyzs[:, 1]  # Negate ys
        xyzs[:, 2] = -xyzs[:, 2]  # Negate zs
        t[2] = -t[2]
        # t[1] = -t[1]
        xyzs = torch.matmul(xyzs, R_rotation.T) + t  # Rotate 
        xyzs[:, 1] = -xyzs[:, 1]  # filp y again
        # xyzs[1] = -xyzs[1]
        # xyzs[2] = -xyzs[2]
        # # xyzs = torch.stack([x, -y, -z], dim=-1).squeeze()  # (num_seed_points, 3)
        # # xyzs[:, 1] = -xyzs[:, 1]  # filp y again
        # xyzs = torch.matmul(xyzs, R.T) + t  # (num_seed_points, 3)
        # if xyzs.is_cuda:
        #     xyzs = xyzs.cpu()
        distances, _ = self.k_nearest_sklearn(xyzs, 3)
                # Initialize scales using 3-nearest neighbors average distance.
        # distances, _ = self.k_nearest_sklearn(xyzs.cpu().numpy(), 3)
        # distances, _ = self.k_nearest_sklearn(xyzs.numpy(), 3)
        # print(self.device)
        distances = torch.from_numpy(distances).to(self.device)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.log(avg_dist.repeat(1, 3))

        # Initialize quats to random.
        num_samples = xyzs.size(0)
        quats = random_quat_tensor(num_samples).to(self.device)

        # Initialize SH features to RGB2SH of the points color.
        dim_sh = num_sh_bases(self.config.sh_degree)
        shs = torch.zeros((num_samples, dim_sh, 3)).float().to(self.device)
        if self.config.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(rgbs)
            shs[:, 1:, 3:] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(rgbs.clamp(min=1e-6, max=1-1e-6), eps=1e-10)
        features_dc = shs[:, 0, :]
        features_rest = shs[:, 1:, :]

        # Initialize opacities to logit(0.3).
        opacities = torch.logit(torch.tensor(0.3).repeat(num_samples, 1)).to(self.device)
        
        # Concatenate the new gaussians to the existing ones.
        # Ensure xyzs is on the same device as self.means
        xyzs = xyzs.to(self.means.device)
        # xyzs = torch.matmul(xyzs, R.T) + t  # (num_seed_points, 3)
        self.means = torch.nn.Parameter(torch.cat([self.means.detach(), xyzs], dim=0))
        self.scales = torch.nn.Parameter(torch.cat([self.scales.detach(), scales], dim=0))
        self.quats = torch.nn.Parameter(torch.cat([self.quats.detach(), quats], dim=0))
        self.opacities = torch.nn.Parameter(torch.cat([self.opacities.detach(), opacities], dim=0))
        self.features_dc = torch.nn.Parameter(torch.cat([self.features_dc.detach(), features_dc], dim=0))
        self.features_rest = torch.nn.Parameter(torch.cat([self.features_rest.detach(), features_rest], dim=0))

        # Add the new parameters to the optimizer.
        for param_group, new_param in self.get_gaussian_param_groups().items():
            optimizer = optimizers.optimizers[param_group]
            old_param = optimizer.param_groups[0]["params"][0]
            param_state = optimizer.state[old_param]
            added_param_shape = (num_samples, *new_param[0].shape[1:])
            if "exp_avg" in param_state:
                param_state["exp_avg"] = torch.cat(
                    [param_state["exp_avg"], torch.zeros(added_param_shape).to(self.device)], dim=0)
            if "exp_avg_sq" in param_state:
                param_state["exp_avg_sq"] = torch.cat(
                    [param_state["exp_avg_sq"], torch.zeros(added_param_shape).to(self.device)], dim=0)

            del optimizer.state[old_param]
            optimizer.state[new_param[0]] = param_state
            optimizer.param_groups[0]["params"] = new_param
            del old_param




    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs_base = super().get_training_callbacks(training_callback_attributes)

        cb_seed = TrainingCallback(
            [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            self.seed_cb,
            args=[
                training_callback_attributes.pipeline,
                training_callback_attributes.optimizers,
            ],
        )
        return [cb_seed] + cbs_base
