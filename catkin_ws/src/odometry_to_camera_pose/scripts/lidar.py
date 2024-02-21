#!/usr/bin/env python

import rospy
import tf
import tf2_ros
import tf2_sensor_msgs
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
# import pcl
# import pcl_ros
import rospy
from sensor_msgs.msg import PointCloud2, PointField, TimeReference
import sensor_msgs.point_cloud2 as pc2
# from geometry_msgs.msg import TransformStamped
import struct  # Add this import at the top of your file

# def create_xyzrgb_cloud_optimized(header, points):
#     fields = [
#         PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
#         PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
#         PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
#         PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
#     ]

#     xyz = points[:, :3]
#     rgb = points[:, 3:6]
#        # Pack RGB values into a single uint32
#     rgb_packed = np.left_shift(rgb[:, 0].astype(np.uint32), 16) | \
#                  np.left_shift(rgb[:, 1].astype(np.uint32), 8) | \
#                  rgb[:, 2].astype(np.uint32)
    
#     # Combine XYZ with RGB into a single array (still separate, for performance)
#     # This step is inherently efficient and should not be the bottleneck

#     # The bottleneck is the conversion to a structured array for ROS message creation
#     # Direct structured array creation with minimal manipulation:
#     xyzrgb_structured = np.core.records.fromarrays([xyz[:, 0], xyz[:, 1], xyz[:, 2], rgb_packed], 
#                                                    names='x, y, z, rgb', 
#                                                    formats='f4, f4, f4, u4')
#     # Create PointCloud2 message
#     cloud = pc2.create_cloud(header, fields, xyzrgb_structured)
#     return cloud



def crop_points_to_fov_optimized(header,lidar_points, img, P_rect_00, RT):
    # Transform all points to camera coordinate system in one go
    
    # Assuming lidar_points is an (N, 3) NumPy array
    lidar_points_homogeneous = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    # Now, lidar_points_homogeneous is (N, 4)

    # Correct multiplication
    cam_points = np.dot(lidar_points_homogeneous, RT.T)

    # Filter out points behind the camera
    cam_points = cam_points[cam_points[:, 2] > 0]
    # print("RT shape:", RT.shape) 
    # print("lidar_points_homogeneous shape:", lidar_points_homogeneous.shape)  # Should be (N, 4)
    RT_adjusted = np.array([
        [0., -1., 0., 0.0],  # Map LiDAR's forward (x) to camera's forward (z)
        [0., 0., -1., 0.0], # Map LiDAR's left (y) to camera's right (x)
        [1., 0., 0., 0.0], # Map LiDAR's up (z) to camera's down (y)
        [.07, .05, -.09, 1.]
        ])
    # Project points onto the image plane

    # cam_points = np.dot(cam_points,RT_adjusted.T).T
    # cam_points = cam_points[:, 2] -0.19
    img_points_homogeneous = np.dot(P_rect_00, cam_points.T).T
    
    img_points = img_points_homogeneous / img_points_homogeneous[:, 2].reshape(-1, 1)
    # img_points = np.dot(img_points,RT_adjusted)
    # img_points[:, 2] = img_points[:, 2] -0.19
    # Filter points within image boundaries
    u, v = img_points[:, 0].astype(int), img_points[:, 1].astype(int)
    valid_indices = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
    u, v = u[valid_indices], v[valid_indices]
    cam_points_valid = cam_points[valid_indices]
    cam_points_valid = np.dot(cam_points_valid,RT_adjusted)
    # Extract RGB values
    rgb = img[v, u]
    # cam_points_valid[:, 2] = cam_points_valid[:, 2] -0.19
    # print("cam_points_valid shape:", cam_points_valid.shape)  # Should be (N, 4)
    # Concatenate XYZ with RGB
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        # PointField for RGB must have datatype=PointField.UINT32, count=1, and offset=12
        # We pack R, G, B values into a single float (this is a common trick)
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    rgb_packed = np.left_shift(rgb[:, 2].astype(np.uint32), 16) | \
                 np.left_shift(rgb[:, 1].astype(np.uint32), 8) | \
                 rgb[:, 0].astype(np.uint32)
    xyzrgb_structured = np.core.records.fromarrays([cam_points_valid[:, 0], cam_points_valid[:, 1], cam_points_valid[:, 2], rgb_packed], 
                                                   names='x, y, z, rgb', 
                                                   formats='f4, f4, f4, u4')
    cloud = pc2.create_cloud(header, fields, xyzrgb_structured)
    # xyzrgb_points = np.hstack((cam_points_valid[:, :3], rgb_values))

    return cloud
# def crop_points_to_fov(lidar_points, img, P_rect_00, RT):
#     filtered_points = []

#     for point in lidar_points:
#         x, y, z = point
#         lidar_point = np.array([x, y, z, 1])
#         cam_point = np.dot(RT, lidar_point)
#         if cam_point[2] <= 0:  # If z <= 0, the point is behind the camera or on the camera plane, so skip it
#             continue
#         img_point = np.dot(P_rect_00, cam_point)
#         img_point /= img_point[2]
       
#         u, v = int(img_point[0]), int(img_point[1])

#         # Check if the point is within the image boundaries
#         if 0 <= u < img.shape[1] and 0 <= v < img.shape[0] :
#             rgb = img[int(v), int(u)]  # Assuming BGR format from OpenCV
#             xyzrgb = np.append(point[:3], rgb)
#             filtered_points.append(xyzrgb)

#     return filtered_points

class PointCloudTransform:
    def __init__(self):
        rospy.init_node('pointcloud_rgb_overlay')

        # self.last_image = None  # Initialize the variable to store the latest image

        # Subscribers
        self.pc_sub = rospy.Subscriber('/velodyne_points', PointCloud2, self.point_cloud_callback)
        self.image_sub = rospy.Subscriber('/zed_node/rgb/image_rect_color', Image, self.image_callback)
        # TF Listener
        self.tf_listener = tf.TransformListener()

        self.bridge = CvBridge()
        self.last_image = None
        self.last_image_time = None  # Timestamp of the last received image
        self.image_received = False  # Flag to indicate a new image has been received


        self.pc_pub = rospy.Publisher('/transformed_points', PointCloud2, queue_size=10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_image = cv_image
            self.image_received = True  
        except Exception as e:
            rospy.logerr("Failed to convert image: %s", e)




    def point_cloud_callback(self, msg):
        if not self.image_received:
            rospy.loginfo("No new image received yet")
            return

        try:
            self.image_received = False
            # Wait for the transformation to be available
            self.tf_listener.waitForTransform('zed2_camera_center', msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform('zed2_camera_center', msg.header.frame_id, rospy.Time(0))
            # print(self.last_image)

            # Read points from the incoming PointCloud2 message
            points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

            # List to hold transformed points
            transformed_points = []

            # For each point in the point cloud, apply the transformation
            for point in points:
                # Apply the transformation (this is a simplification, actual transformation involves rotation as well)
                transformed_x = point[0] + trans[0]
                transformed_y = point[1] + trans[1]
                transformed_z = point[2] + trans[2]

                # Append the transformed point to the list
                transformed_points.append([transformed_x, transformed_y, transformed_z])
            
                  # Your provided P values
            P_values = [532.9841918945312, 0.0, 550.3889770507812, 0.0, 
                        0.0, 532.9841918945312, 303.86602783203125, 0.0, 
                        0.0, 0.0, 1.0, 0.0]

            # Reshape P_values into a 3x4 matrix
            P_rect_00 = np.array(P_values).reshape(3, 4)
            R_rect_00 = np.eye(4)  # Identity matrix for simplicity

            RT = np.array([[0., -1., 0., -0.0],
                           [0., 0., -1., -0.0],
                           [1., 0., 0., -0.0],
                           [0., 0., 0., 1.]])
            # RT_adjusted = np.array([
            #                         [0., 0., 1., -0.01],  # Map LiDAR's forward (x) to camera's forward (z)
            #                         [-1., 0., 0., -0.06], # Map LiDAR's left (y) to camera's right (x)
            #                         [0., -1., 0., -0.05], # Map LiDAR's up (z) to camera's down (y)
            #                         [0., 0., 0., 1.]
            #                     ])

            xyz_points = []
            rgb_colors = []
            lidar_points = np.array(transformed_points)
            xyzrgb_points = crop_points_to_fov_optimized(msg.header,lidar_points, self.last_image, P_rect_00, RT)
            xyzrgb_points.header.stamp = rospy.Time.now()
            # transformed_cloud_msg = create_xyzrgb_cloud_optimized(, xyzrgb_points)
            # transformed_cloud_msg.header.frame_id = 'zed2_camera_center'
          
            
            self.pc_pub.publish(xyzrgb_points)
            
            # Update the header to reflect the new frame
           

            # Publish the transformed point cloud
            # self.pc_pub.publish(transformed_cloud_msg)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(e)



    def process_and_overlay(self, pc):
       

        pass

if __name__ == '__main__':
    try:

        PointCloudTransform()
   

        rospy.spin()
    except rospy.ROSInterruptException:
        pass

