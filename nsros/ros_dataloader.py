# Code adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/data/utils/dataloaders.py

"""
Defines the ROSDataloader object that subscribes to pose and images topics,
and populates an image tensor and Cameras object with values from these topics.
Image and pose pairs are added at a prescribed frequency and intermediary images
are discarded (could be used for evaluation down the line).
"""
import time
import warnings
from typing import Union
# import torch
import numpy as np
import scipy.spatial.transform as transform
from rich.console import Console
import torch
from torch.utils.data.dataloader import DataLoader

from nerfstudio.process_data.colmap_utils import qvec2rotmat
import nerfstudio.utils.poses as pose_utils

from nsros.ros_dataset import ROSDataset

import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, PoseArray
from message_filters import TimeSynchronizer, ApproximateTimeSynchronizer,Subscriber
from scipy.spatial.transform import Rotation 

CONSOLE = Console(width=120)

# Suppress a warning from torch.tensorbuffer about copying that
# does not apply in this case.
warnings.filterwarnings("ignore", "The given buffer")

def ros_pose_to_homogenous(pose_message: PoseStamped):
    """
    Converts a ROS2 Pose message to a 4x4 homogenous transformation matrix
    as a torch tensor (half precision).
    """
    quat = pose_message.pose.orientation
    pose = pose_message.pose.position

    R = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    t = torch.Tensor([pose.x, pose.y, pose.z])

    T = torch.eye(4)
    T[:3, :3] = torch.from_numpy(R)
    T[:3, 3] = t
    return T.to(dtype=torch.float32)

def ros_pose_to_nerfstudio(pose: PoseStamped, static_transform=None):
    hom_pose =ros_pose_to_homogenous(pose)
    T_ns = hom_pose[:, [1, 2, 0, 3]]
    T_ns[:, [0, 2]] *= -1
    return T_ns[:3, :]
    # """
    # Takes a ROS Pose message and converts it to the
    # 3x4 transform format used by nerfstudio.
    # """
    # pose_msg = pose.pose
    # quat = np.array(
    #     [
    #         pose_msg.orientation.w,
    #         pose_msg.orientation.x,
    #         pose_msg.orientation.y,
    #         pose_msg.orientation.z,
    #     ],
    # )

    
    # if static_transform is not None:
    #     posi = torch.tensor([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    #     R = torch.tensor(qvec2rotmat(quat))
    #     T = torch.cat([R, posi.unsqueeze(-1)], dim=-1)
    #     T = T.to(dtype=torch.float32)
    #     # print(T)
    #     print("-------------------------")
    #     T = pose_utils.multiply(T, static_transform)
    #     T= T[:, [1, 2, 0, 3]]
    #     T[:, [0, 2]] *= -1
    #     # print("-------------------------")
    #     # print(T)
    #     # print("-------------------------")
    #     # T2 = torch.zeros(3, 4)
    #     # R1 = transform.Rotation.from_euler("x", 0, degrees=True).as_matrix()
    #     # R2 = transform.Rotation.from_euler("z", 0, degrees=True).as_matrix()
    #     # R3 = transform.Rotation.from_euler("y", 0, degrees=True).as_matrix()
    #     # R = torch.from_numpy(R3 @ R2 @ R1)
    #     # T2[:, :3] = R
    #     # T = pose_utils.multiply(T2, T)
    #     print(T[:3, :])

    

    #     return T[:3, :].to(dtype=torch.float32)


class ROSDataloader(DataLoader):
    """
    Creates batches of the dataset return type. In this case of nerfstudio this means
    that we are returning batches of full images, which then are sampled using a
    PixelSampler. For this class the image batches are progressively growing as
    more images are recieved from ROS, and stored in a pytorch tensor.

    Args:
        dataset: Dataset to sample from.
        publish_posearray: publish a PoseArray to a ROS topic that tracks the poses of the
            images that have been added to the training set.
        data_update_freq: Frequency (wall clock) that images are added to the training
            data tensors. If this value is less than the frequency of the topics to which
            this dataloader subscribes (pose and images) then this subsamples the ROS data.
            Otherwise, if the value is larger than the ROS topic rates then every pair of
            messages is added to the training bag.
        device: Device to perform computation.
    """

    dataset: ROSDataset
    # xyzrgb_dataset: ROSXYZRGBDataset
    # print(torch.device)
    def __init__(
        self,
        dataset: ROSDataset,
        data_update_freq: float,
        device: Union[torch.device, str] = "cpu",
        publish_posearray: bool =True,
        listen_depth: bool = True,
        **kwargs,
      
    ):
        # This is mostly a parameter placeholder, and manages the cameras
        self.dataset = dataset
        self.update_period = 1 / data_update_freq
        self.device = device
        self.listen_depth = listen_depth
        self.num_images = len(self.dataset)
        self.H = self.dataset.image_height
        self.W = self.dataset.image_width
        self.n_channels = 3

        # Tracking ros updates
        self.current_idx = 0
        self.updated = True
        
        self.last_update_t = time.perf_counter()
        self.publish_posearray = publish_posearray
        self.poselist = []

        self.coord_st = torch.zeros(3, 4)
        R1 = transform.Rotation.from_euler("x", 180, degrees=True).as_matrix()
        R2 = transform.Rotation.from_euler("z", 0, degrees=True).as_matrix()
        R = torch.from_numpy(R2 @ R1)
        self.coord_st[:, :3] = R

        # Keep it in the format so that it makes it look more like a
        # regular data loader.
        self.data_dict = {
            "image": self.dataset.image_tensor,
            "image_idx": self.dataset.image_indices,
        }

        super().__init__(dataset=dataset, **kwargs)

        # All of the ROS CODE
        rospy.init_node("nsros_dataloader", anonymous=True)
        print("========================Reading============================")
        print("Image sub")
        self.image_sub = Subscriber(self.dataset.image_topic_name, Image)
        print("PointCloud2 sub")
        self.lidar_sub = Subscriber(self.dataset.point_cloud_topic_name,PointCloud2)
        print("pose_sub sub")
        self.pose_sub = Subscriber(self.dataset.pose_topic_name, PoseStamped)

        
        # self.ts = TimeSynchronizer([self.image_sub, self.pose_sub], 100)
        print("=======================TimeSynchronizer==========================")
        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.pose_sub,self.lidar_sub], 100, 0.05)
        # self.ts = TimeSynchronizer([self.image_sub, self.pose_sub,self.lidar_sub], 100)
# self.lidar_sub
        print("=======================After Sync==========================")
        # print(self.ts_image_pose_callback)
        # self.ts_image_pose_callback 
        self.ts.registerCallback(self.ts_image_pose_callback)
        
        self.posearray_pub = rospy.Publisher("training_poses", PoseArray, queue_size=1)


    def msg_status(self, num_to_start):
        """
        Check if any image-pose pairs have been successfully streamed from
        ROS, and return True if so.
        """
        return self.current_idx >= (num_to_start - 1)

    def ts_image_pose_callback(self, image: Image, pose: PoseStamped,cloud_msg: PointCloud2):
        """
        The callback triggered when time synchronized image and pose messages
        are published on the topics specifed in the config JSON passed to
        the ROSDataParser.
        """
        # print("-------------------------------started---------------")
        now = time.perf_counter()
        if (
            now - self.last_update_t > self.update_period
            and self.current_idx < self.num_images
        ):
            # ----------------- Handling the IMAGE ----------------
            # Load the image message directly into the torch
            print("-------------image index: ",self.current_idx,"----------------------")
            im_tensor = torch.frombuffer(image.data, dtype=torch.uint8).reshape(
                self.H, self.W, -1
            )
            im_tensor = im_tensor.to(dtype=torch.float32) / 255.0
            # Convert BGR -> RGB (this adds an extra copy, and might be able to
            # skip if we do something fancy with the reshape above)
           
            # im_tensor = im_tensor.flip([-1])

            # COPY the image data into the data tensor
            im_tensor_rgb = im_tensor[:, :, :3]  # Keep only the first 3 channels (RGB)
            im_tensor_rgb = im_tensor_rgb[:, :, [2, 1, 0]]  # Reorder channels from BGR to RGB
            # im_tensor_rgb = im_tensor[:, :, [2, 1, 0]]  # Rearrange the channels

            # image_np = im_tensor_rgb.cpu().detach().numpy()
            # plt.imshow(image_np)
            # plt.axis('off')  # Turn off axis numbers and ticks
            # plt.show()
            self.dataset.image_tensor[self.current_idx] = im_tensor_rgb

            # ----------------- Handling the POSE ----------------
            c2w = ros_pose_to_nerfstudio(pose, static_transform=self.coord_st)
            device = self.dataset.cameras.device
            c2w = c2w.to(device)
            self.dataset.cameras.camera_to_worlds[self.current_idx] = c2w

            if self.publish_posearray:
                self.poselist.append(pose.pose)
                pa = PoseArray(poses=self.poselist)
                pa.header.frame_id = "map"
                self.posearray_pub.publish(pa)
            # print("-------------image pose: ",self.poselist,"----------------------")
            self.dataset.updated_indices.append(self.current_idx)

            #-------------------------------Handling Lidar------------------------------
            cloud_points = list(pc2.read_points(cloud_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        
            # Separate XYZ coordinates and RGB colors
            xyz = [(x, y, z) for x, y, z, _ in cloud_points]
            rgbs = [rgb for _, _, _, rgb in cloud_points]
            
            # Convert RGB floats to integer values, then unpack them into separate R, G, B channels
            rgb_ints = np.array(rgbs, dtype=np.uint32)
            r = ((rgb_ints >> 16) & 0xFF).astype(np.int64)  # Convert to np.int64
            g = ((rgb_ints >> 8) & 0xFF).astype(np.int64)   # Convert to np.int64
            b = (rgb_ints & 0xFF).astype(np.int64)          # Convert to np.int64
            
            # Convert XYZ and RGB data into PyTorch tensors for further processing
            xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
            rgb_tensor = torch.stack((
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(g, dtype=torch.float32),
                torch.tensor(b, dtype=torch.float32)
            ), dim=1)
            device = self.dataset.cameras.device
            xyzrgb_tensor = torch.cat((xyz_tensor, rgb_tensor / 255.0), dim=1)  # Normalize RGB to [0, 1]
            xyzrgb_tensor = xyzrgb_tensor.to(device)
            self.dataset.update_point_cloud(self.current_idx, xyzrgb_tensor)
            # print("saved Lidar points")
            #---------------------------- end of lidar process------------------------------------
            self.updated = True
            self.current_idx += 1
            self.last_update_t = now

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def _get_updated_batch(self):
        batch = {}
        for k, v in self.data_dict.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v[: self.current_idx, ...]
        return batch

    def __iter__(self):
        while True:
            if self.updated:
                self.batch = self._get_updated_batch()
                self.updated = False

            batch = self.batch
            yield batch

    # def xyzrgb_callback(self, cloud_msg: PointCloud2):
    #     """
    #     Callback for processing XYZRGB point cloud data. This function extracts XYZ coordinates and RGB colors,
    #     converts them into a suitable format for processing or neural network input.
    #     """
    #     # Convert the ROS PointCloud2 message to a list of (x, y, z, rgb) tuples
    #     cloud_points = list(pc2.read_points(cloud_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
        
    #     # Separate XYZ coordinates and RGB colors
    #     xyz = [(x, y, z) for x, y, z, _ in cloud_points]
    #     rgbs = [rgb for _, _, _, rgb in cloud_points]
        
    #     # Convert RGB floats to integer values, then unpack them into separate R, G, B channels
    #     rgb_ints = np.array(rgbs, dtype=np.uint32)
    #     r = (rgb_ints >> 16) & 0xFF
    #     g = (rgb_ints >> 8) & 0xFF
    #     b = rgb_ints & 0xFF
        
    #     # Convert XYZ and RGB data into PyTorch tensors for further processing
    #     xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
    #     rgb_tensor = torch.stack((torch.tensor(r, dtype=torch.float32),
    #                             torch.tensor(g, dtype=torch.float32),
    #                             torch.tensor(b, dtype=torch.float32)), dim=1)
        
    #     # Example: Combine XYZ and RGB data into a single tensor if needed
   
    #     # Note: Ensure the combined data is in the format expected by your model or processing pipeline
    #     xyzrgb_tensor = torch.cat((xyz_tensor, rgb_tensor / 255.0), dim=1)  # Normalize RGB to [0, 1]
    #     print("------------------------Data come here--------------------")
    #     # Now you have the xyzrgb_tensor which you can use as needed, for example, as input to a neural network
    #     # Or further process it depending on your application's requirements
        
    #     # Example of using the tensor
    #     # Here you could apply some transformations, use it as input to a model, etc.
    #     # This part is highly dependent on your specific application