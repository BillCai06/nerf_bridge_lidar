# Code adapted from Nerfstudio
# https://github.com/nerfstudio-project/nerfstudio/blob/df784e96e7979aaa4320284c087d7036dce67c28/nerfstudio/data/utils/dataloaders.py

"""
Defines the ROSDataloader object that subscribes to pose and images topics,
and populates an image tensor and Cameras object with values from these topics.
Image and pose pairs are added at a prescribed frequency and intermediary images
are discarded (could be used for evaluation down the line).
"""
import time
import threading
import warnings
from typing import Union

from rich.console import Console
import torch
import numpy as np
from torchvision.transforms import Resize
from torch.utils.data.dataloader import DataLoader

import nerfbridge.pose_utils as pose_utils
from nerfbridge.ros_dataset import ROSDataset, ROSDepthDataset

import rclpy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer, Subscriber
from cv_bridge import CvBridge

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from nerfstudio.process_data.colmap_utils import qvec2rotmat
import scipy.spatial.transform as transform
from scipy.spatial.transform import Rotation 



CONSOLE = Console(width=120)

# Suppress a warning from torch.tensorbuffer about copyhttps://github.com/BillCai06/nerf_bridge_lidar.giting that
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
        slam_method: string that determines which type of pose topic to subscribe to and
            what coordinate transforms to use when handling the poses. Currently, only
            "cuvslam" is supported.
        topic_sync: use "exact" when your slam algorithm matches pose and image time stamps,
            and use "approx" when it does not.
        topic_slop: if using approximate time synchronization, then this float determines
            the slop in seconds that is allowable between images and poses.
        device: Device to perform computation.
    """

    dataset: ROSDataset

    def __init__(
        self,
        dataset: ROSDataset,
        data_update_freq: float,
        slam_method: str,
        topic_sync: str,
        topic_slop: float,
        use_compressed_rgb: bool,
        device: Union[torch.device, str] = "cpu",
        publish_posearray: bool = True,
        **kwargs,
    ):
        # This is mostly a parameter placeholder, and manages the cameras
        self.dataset = dataset
        # Image meta data
        self.device = device
        self.num_images = len(self.dataset)
        self.H = self.dataset.image_height
        self.W = self.dataset.image_width
        self.n_channels = 3

        # Tracking ros updates
        self.current_idx = 0
        self.updated = True
        self.update_period = 1 / data_update_freq
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

        # # Flag for depth training, add depth tensor to data.
        # self.listen_depth = False
        # if isinstance(self.dataset, ROSDepthDataset):
        #     self.data_dict["depth_image"] = self.dataset.depth_tensor
        #     # For resizing depth image to match color image.
        #     self.depth_transform = Resize((self.H, self.W))
        #     self.listen_depth = True

        super().__init__(dataset=dataset, **kwargs)

        self.bridge = CvBridge()
        self.slam_method = slam_method
        self.use_compressed_rgb = use_compressed_rgb

        # Initializing ROS2
        rclpy.init()
        self.node = rclpy.create_node("nerf_bridge_node")

        self.health_pub = self.node.create_publisher(String, "dataloader_health", 10)

        # Setting up ROS2 message_filter TimeSynchronzier
        self.subs = []
        if self.use_compressed_rgb:
            self.subs.append(
                Subscriber(
                    self.node,
                    CompressedImage,
                    self.dataset.image_topic_name,
                )
            )
        else:
            self.subs.append(
                Subscriber(
                    self.node,
                    Image,
                    self.dataset.image_topic_name,
                )
            )

        if slam_method == "cuvslam":
            self.subs.append(
                Subscriber(self.node, Odometry, self.dataset.pose_topic_name)
            )
        elif slam_method == "orbslam3":
            self.subs.append(
                Subscriber(self.node, PoseStamped, self.dataset.pose_topic_name)
            )
        elif slam_method == "mocap":
            self.subs.append(
                Subscriber(self.node, PoseStamped, self.dataset.pose_topic_name)
            )
        else:
            raise NameError(
                "Unsupported SLAM algorithm. Must be one of {cuvslam, orbslam3}"
            )
        
        self.subs.append(
            Subscriber(self.node, PointCloud2, self.dataset.lidar_topic_name)
        )

        # if self.listen_depth:
        #     self.subs.append(
        #         Subscriber(self.node, Image, self.dataset.depth_topic_name)
        #     )

        if topic_sync == "approx":
            self.ts = ApproximateTimeSynchronizer(self.subs, 40, topic_slop)
        elif topic_sync == "exact":
            self.ts = TimeSynchronizer(self.subs, 40)
        else:
            raise NameError(
                "Unsupported topic sync method. Must be one of {approx, exact}."
            )

        self.ts.registerCallback(self.ts_callback)
        self.posearray_pub = self.node.create_publisher(PoseArray, 'training_poses', 1)

        # Start a thread for processing the callbacks
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True
        )
        self.ros_thread.start()

    def msg_status(self, num_to_start):
        """
        Check if any image-pose pairs have been successfully streamed from
        ROS, and return True if so.
        """
        return self.current_idx >= (num_to_start - 1)

    def ts_callback(self, image: Image, pose: PoseStamped, cloud_msg: PointCloud2):
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
            # print("-------------image index: ",self.current_idx,"----------------------")
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

    # def lidar_callback(self, lidar_msg: pc2.PointCloud2):
    #     cloud_points = list(pc2.read_points(lidar_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
    
    #     # Separate XYZ coordinates and RGB colors
    #     xyz = [(x, y, z) for x, y, z, _ in cloud_points]
    #     rgbs = [rgb for _, _, _, rgb in cloud_points]
        
    #     # Convert RGB floats to integer values, then unpack them into separate R, G, B channels
    #     rgb_ints = np.array(rgbs, dtype=np.uint32)
    #     r = ((rgb_ints >> 16) & 0xFF).astype(np.int64)  # Convert to np.int64
    #     g = ((rgb_ints >> 8) & 0xFF).astype(np.int64)   # Convert to np.int64
    #     b = (rgb_ints & 0xFF).astype(np.int64)          # Convert to np.int64
        
    #     # Convert XYZ and RGB data into PyTorch tensors for further processing
    #     xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
    #     rgb_tensor = torch.stack((
    #         torch.tensor(r, dtype=torch.float32),
    #         torch.tensor(g, dtype=torch.float32),
    #         torch.tensor(b, dtype=torch.float32)
    #     ), dim=1)
    #     device = self.dataset.cameras.device
    #     xyzrgb_tensor = torch.cat((xyz_tensor, rgb_tensor / 255.0), dim=1)  # Normalize RGB to [0, 1]
    #     xyzrgb_tensor = xyzrgb_tensor.to(device)
    #     self.dataset.update_point_cloud(self.current_idx, xyzrgb_tensor)

    # def image_callback(self, image: Image | CompressedImage):
    #     """
    #     Callback for processing RGB Image Messages, and adding them to the
    #     dataset for training.
    #     """
    #     # Load the image message directly into the torch
    #     if self.use_compressed_rgb:
    #         im_cv = self.bridge.compressed_imgmsg_to_cv2(image)
    #         im_tensor = torch.from_numpy(im_cv).to(dtype=torch.float32) / 255.0
    #         im_tensor = torch.flip(im_tensor, [-1])
    #     else:
    #         im_cv = self.bridge.imgmsg_to_cv2(image, image.encoding)
    #         im_tensor = torch.from_numpy(im_cv).to(dtype=torch.float32) / 255.0

    #     # COPY the image data into the data tensor
    #     im_tensor_rgb = im_tensor[:, :, :3]  # Keep only the first 3 channels (RGB)
    #     im_tensor_rgb = im_tensor_rgb[:, :, [2, 1, 0]]  # Reorder channels from BGR to RGB
    #     # im_tensor_rgb = im_tensor[:, :, [2, 1, 0]]  # Rearrange the channels

    #     self.dataset.image_tensor[self.current_idx] = im_tensor_rgb

    # def pose_callback(self, pose: PoseStamped | Odometry):
    #     """
    #     Callback for Pose messages. Extracts pose, converts it to Nerfstudio coordinate
    #     convention, and inserts it into the Cameras object.
    #     """
    #     if self.slam_method == "cuvslam":
    #         # Odometry Message
    #         hom_pose = pose_utils.ros_pose_to_homogenous(pose.pose)
    #         c2w = pose_utils.cuvslam_to_nerfstudio(hom_pose)
    #     elif self.slam_method == "orbslam3":
    #         # PoseStamped Message
    #         hom_pose = pose_utils.ros_pose_to_homogenous(pose)
    #         c2w = pose_utils.orbslam3_to_nerfstudio(hom_pose)
    #     elif self.slam_method == "mocap":
    #         # PoseStamped Message
    #         hom_pose = pose_utils.ros_pose_to_homogenous(pose)
    #         c2w = pose_utils.mocap_to_nerfstudio(hom_pose)
    #         self.publish_posearray=True
    #     else:
    #         raise NameError("Unknown SLAM Method!")

    #     # Scale Pose to
    #     c2w[:3, 3] *= self.dataset.scale_factor

    #     # Insert in Cameras
    #     device = self.dataset.cameras.device
    #     self.dataset.cameras.camera_to_worlds[self.current_idx] = c2w.to(device)

    #     if self.publish_posearray:
    #         self.poselist.append(pose.pose)
    #         pa = PoseArray(poses=self.poselist)
    #         pa.header.frame_id = "map"
    #         self.posearray_pub.publish(pa)
    #     # print("-------------image pose: ",self.poselist,"----------------------")
    #     self.dataset.updated_indices.append(self.current_idx)

    # def depth_callback(self, depth: Image):
    #     """
    #     Callback for processing Depth Image messages. Similar to RGB image handling,
    #     but also rescales the depth to the appropriate value.
    #     """
    #     depth_cv = self.bridge.imgmsg_to_cv2(depth, depth.encoding)
    #     depth_tensor = torch.from_numpy(depth_cv.astype("float32")).to(
    #         dtype=torch.float32
    #     )

    #     aggregate_scale = (
    #         self.dataset.scene_scale_factor * self.dataset.depth_scale_factor
    #     )

    #     self.dataset.depth_tensor[self.current_idx] = (
    #         depth_tensor.unsqueeze(-1) * aggregate_scale
    #     )

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
