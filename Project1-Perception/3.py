import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
# import time
from jsk_recognition_msgs.msg import BoundingBox
from sklearn.cluster import DBSCAN, OPTICS
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import open3d as o3d

def max_min_list(current_points: list):
    x_list = [point[0] for point in current_points]
    y_list = [point[1] for point in current_points]
    z_list = [point[2] for point in current_points]
    max_x = max(x_list)
    max_y = max(y_list)
    max_z = max(z_list)
    min_x = min(x_list)
    min_y = min(y_list)
    min_z = min(z_list)
    return max_x, max_y, max_z, min_x, min_y, min_z



def xyz_dimension_xyz(point: np.ndarray, labels: np.ndarray, box_array: BoundingBoxArray):
    """
    Parameter
    ---------
    point: lidar point cloud
    labels: after clustering's labels
    """
    boxArray = box_array
    max_label = np.amax(labels)
    res_table = np.zeros((max_label+1, 6))
    for label in range(max_label + 1):
        current_points = [point[index, :4] for index, j in enumerate(labels) if j == label]
        max_x, max_y, max_z, min_x, min_y, min_z = max_min_list(current_points)
        center_x = 0.5*(max_x+min_x)
        center_y = 0.5*(max_y+min_y)
        center_z = 0.5*(max_z+min_z)
        dim_x = max_x - min_x
        dim_y = max_y - min_y
        dim_z = max_z - min_z
        res_table[label, :] = np.array([center_x, center_y, center_z, dim_x, dim_y, dim_z])
        box = BoundingBox()
        box.header.frame_id = 'lidar_top'
        box.pose.orientation.w = 1.0
        box.pose.position.x = center_x
        box.pose.position.y = center_y
        box.pose.position.z = center_z
        box.dimensions.x = dim_x
        box.dimensions.y = dim_y
        box.dimensions.z = dim_z
        boxArray.boxes.append(box)
    # print(len(boxArray.boxes))
    return boxArray



def callback(data, pub):
    # time_now = time.time()
    pointcloud = pc2.read_points(data, skip_nans=True)
    list_data = list(pointcloud)
    num_data = len(list_data)
    point_numpy = np.ones((num_data, 4))
    # print(f"test:{list_data[num_data-1]}")
    for index in range(num_data):
        # print(f"test:{list_data[index]}")
        point_numpy[index, 0] = list_data[index][0]
        point_numpy[index, 1] = list_data[index][1]
        point_numpy[index, 2] = list_data[index][2]
        point_numpy[index, 3] = list_data[index][3]
    # print(f"process time:{time.time()-time_now}")

    # remove ground
    points = point_numpy
    mask = np.ones(points.shape[0], dtype=bool)
    mask = np.logical_and(mask, points[:, 2] > -0.8, np.sqrt(points[:, 0]**2+points[:, 1]**2)<=50)
    points = points[mask, :]

    # downsample
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
    pcd_ds = pcd.voxel_down_sample(voxel_size=3.0)
    points_new = np.asarray(pcd_ds.points)
    points_new = np.hstack((points_new, 50*np.ones((points_new.shape[0], 1))))
    # print(points_new.shape)

    point_numpy = points_new
    boxArray = BoundingBoxArray()
    boxArray.header.frame_id = 'lidar_top'

    clustering = DBSCAN(eps=3.5, min_samples=10).fit(point_numpy[:, :3])
    boxArray = xyz_dimension_xyz(point_numpy, clustering.labels_, boxArray)

    
    pub.publish(boxArray)
    # while not rospy.is_shutdown():
    #    pub.publish()
    #    rate.sleep()
    # print(f"time:{time.time()-time_now}")

def listener():
    rospy.init_node('task3')
    rate = rospy.Rate(10)
    pub = rospy.Publisher('/me5413/bbox', BoundingBoxArray, queue_size=1)
    cloud  = rospy.Subscriber(name="/me5413/lidar_top",data_class=PointCloud2,callback=callback, callback_args=pub, queue_size=1)
    rospy.spin()

if __name__=='__main__':
    listener()

