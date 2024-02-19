from typing import Union

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


class ROSDataset(InputDataset):
    """
    This is a tensor dataset that keeps track of all of the data streamed by ROS.
    It's main purpose is to conform to the already defined workflow of nerfstudio:
        (dataparser -> inputdataset -> dataloader).

    In reality we could just store everything directly in ROSDataloader, but this
    would require rewritting more code than its worth.

    Images are tracked in self.image_tensor with uninitialized images set to
    all white (hence torch.ones).
    Poses are stored in self.cameras.camera_to_worlds as 3x4 transformation tensors.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "image_topic" in dataparser_outputs.metadata.keys()
            and "pose_topic" in dataparser_outputs.metadata.keys()
            and "num_images" in dataparser_outputs.metadata.keys()
            and "point_cloud_topic" in dataparser_outputs.metadata.keys()
        )

        self.image_topic_name = self.metadata["image_topic"]
        self.pose_topic_name = self.metadata["pose_topic"]
        self.num_images = self.metadata["num_images"]
        self.point_cloud_topic_name = self.metadata["point_cloud_topic"]
        print(self.point_cloud_topic_name)
        # Initialize the tensor to hold XYZRGB data: Assuming the point cloud's maximum size for pre-allocation
        # You might want to adjust the size based on your expected point cloud size or make it dynamic
        self.max_points = 10000  # Example size, adjust based on your application's needs
        self.xyzrgb_tensor = torch.zeros(self.num_images, self.max_points,  6,  dtype=torch.float32)

       
        assert self.num_images > 0
        self.image_height = self.metadata["image_height"]
        self.image_width = self.metadata["image_width"]
        self.device = device
        self.cameras = self.cameras.to(device=self.device)

        self.image_tensor = torch.ones(
            self.num_images, self.image_height, self.image_width, 3, dtype=torch.float32
        )
        self.image_indices = torch.arange(self.num_images)

        self.updated_indices = []

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx: int):
        """
        This returns the data as a dictionary which is not actually how it is
        accessed in the dataloader, but we allow this as well so that we do not
        have to rewrite the several downstream functions.
        """
        
        data = {"image_idx": idx, "image": self.image_tensor[idx],"xyzrgb": self.xyzrgb_tensor[idx]}
        # data["xyzrgb"] = 
        return data


    def update_point_cloud(self, idx: int, xyzrgb_data: torch.Tensor):
        """
        Update the stored XYZRGB data at the specified index, truncating or padding the data
        as necessary to fit the pre-allocated tensor size.

        Args:
            idx: Index of the image/point cloud pair in the dataset.
            xyzrgb_data: A tensor containing the XYZRGB data for the given index.
        """
     
        if idx < self.num_images:
          
            # Determine whether to truncate or pad the xyzrgb_data
            if xyzrgb_data.shape[0] > self.max_points:
                # Truncate the xyzrgb_data if it has more points than max_points
                truncated_data = xyzrgb_data[:self.max_points, :]
                self.xyzrgb_tensor[idx] = truncated_data
            elif xyzrgb_data.shape[0] < self.max_points:
                # Pad the xyzrgb_data with zeros if it has fewer points than max_points
                padded_data = torch.zeros((self.max_points, xyzrgb_data.shape[1]), dtype=xyzrgb_data.dtype, device=xyzrgb_data.device)
                padded_data[:xyzrgb_data.shape[0], :] = xyzrgb_data
                self.xyzrgb_tensor[idx] = padded_data
                print("Feed lidar data to idx with padding",idx," with shape:",self.xyzrgb_tensor[idx].shape[0])

            else:
                # Directly assign the data if it matches the expected size
                self.xyzrgb_tensor[idx] = xyzrgb_data
        else:
            raise IndexError("Index out of bounds for the dataset.")

# class ROSXYZRGBDataset(ROSDataset):
#     """
#     A tensor dataset for managing XYZRGB point cloud data streamed by ROS.
#     This dataset conforms to the workflow of nerfstudio (dataparser -> inputdataset -> dataloader)
#     and extends the ROSDataset with support for point cloud data.

#     The XYZRGB point cloud data is tracked in self.xyzrgb_tensor. Poses are stored in
#     self.cameras.camera_to_worlds as 3x4 transformation tensors.

#     Args:
#         dataparser_outputs: Description of where and how to read input images and point clouds.
#         scale_factor: The scaling factor for the dataparser outputs.
#         device: The device on which tensors should be allocated (e.g., 'cpu', 'cuda').
#     """

#     # def __init__(
#     #     self,
#     #     dataparser_outputs: DataparserOutputs,
#     #     scale_factor: float = 1.0,
#     #     # device: Union[torch.device, str] = "cpu",
#     #     ):

#     #     super().__init__(dataparser_outputs, scale_factor)
#         # print("HERE ___________________________________________________________________")
#         # assert "depth_topic" in dataparser_outputs.metadata.keys()
        
      

    
    
