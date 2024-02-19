import rospy
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, PoseStamped




def odometry_callback(msg):
    global tf_buffer
    # Create a PoseStamped from the Odometry message
    pose_stamped = PoseStamped()
    pose_stamped.header = msg.header
    pose_stamped.pose = msg.pose.pose
    
    try:

        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = rospy.Time.now()
        transform_stamped.header.frame_id = "map"  # Adjust according to your needs
        transform_stamped.child_frame_id = "base_link"  # This is just for reference; not used in the transformation
        transform_stamped.transform.translation.x = 0.0
        transform_stamped.transform.translation.y = 0.0
        transform_stamped.transform.translation.z = 0.53
        transform_stamped.transform.rotation.x = 0.0  # No rotation
        transform_stamped.transform.rotation.y = 0.0
        transform_stamped.transform.rotation.z = 0.0
        transform_stamped.transform.rotation.w = 1.0  # Represents no rotation
        transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform_stamped)
        # camera_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform2)
        transformed_pose.header.stamp = rospy.Time.now()
        # Publish the camera pose
        camera_pose_pub.publish(transformed_pose)
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
        rospy.logwarn('Error transforming from base_link to camera_link: %s' % str(ex))

if __name__ == '__main__':
    rospy.init_node('odometry_to_camera_pose')
    
    # TF2 buffer and listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    # Subscriber to the /odometry/filtered topic
    rospy.Subscriber('/odometry/filtered', Odometry, odometry_callback)
    
    # Publisher for the camera pose
    camera_pose_pub = rospy.Publisher('/pose', PoseStamped, queue_size=1)
    
    rospy.spin()