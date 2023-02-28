import numpy as np 
import os
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import rospy
from std_msgs.msg import Header
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
        box.header.frame_id = 'map'
        box.pose.orientation.w = 1.0
        box.pose.position.x = center_x
        box.pose.position.y = center_y
        box.pose.position.z = center_z
        box.dimensions.x = dim_x
        box.dimensions.y = dim_y
        box.dimensions.z = dim_z
        boxArray.boxes.append(box)

    return boxArray




root_path = ""
# frame1:
file_data1 = np.fromfile(os.path.join(root_path, 'frame1.pcd.bin'), dtype=np.float32)
points1 = file_data1.reshape((-1, 5))[:, :4]
# frame2:
file_data2 = np.fromfile(os.path.join(root_path, 'frame2.pcd.bin'), dtype=np.float32)
points2 = file_data2.reshape((-1, 5))[:, :4]
# frame3:
file_data3 = np.fromfile(os.path.join(root_path, 'frame3.pcd.bin'), dtype=np.float32)
points3 = file_data3.reshape((-1, 5))[:, :4]
# frame4:
file_data4 = np.fromfile(os.path.join(root_path, 'frame4.pcd.bin'), dtype=np.float32)
points4 = file_data4.reshape((-1, 5))[:, :4]
# frame5:
file_data5 = np.fromfile(os.path.join(root_path, 'frame5.pcd.bin'), dtype=np.float32)
points5 = file_data5.reshape((-1, 5))[:, :4]
# frame6:
file_data6 = np.fromfile(os.path.join(root_path, 'frame6.pcd.bin'), dtype=np.float32)
points6 = file_data6.reshape((-1, 5))[:, :4]
# frame7:
file_data7 = np.fromfile(os.path.join(root_path, 'frame7.pcd.bin'), dtype=np.float32)
points7 = file_data7.reshape((-1, 5))[:, :4]
# frame8:
file_data8 = np.fromfile(os.path.join(root_path, 'frame8.pcd.bin'), dtype=np.float32)
points8 = file_data8.reshape((-1, 5))[:, :4]
# frame9:
file_data9 = np.fromfile(os.path.join(root_path, 'frame9.pcd.bin'), dtype=np.float32)
points9 = file_data9.reshape((-1, 5))[:, :4]
# frame10:
file_data10 = np.fromfile(os.path.join(root_path, 'frame10.pcd.bin'), dtype=np.float32)
points0 = file_data10.reshape((-1, 5))[:, :4]



def visualize1cloud(points: np.ndarray, eps: float=0.6, min_sample: int=6):
    """
    """
    boxArray = BoundingBoxArray()
    boxArray.header.frame_id = 'map'

    fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          # PointField('rgb', 12, PointField.UINT32, 1),
          PointField('rgba', 12, PointField.FLOAT32, 1),
          ]
    header = Header()
    header.frame_id = "map"

    pc_msg = pc2.create_cloud(header, fields, points)
    clustering = DBSCAN(eps=eps, min_samples=min_sample).fit(points[:, :3])
    # clustering = OPTICS(min_samples=100, xi=0.05).fit(points[:,:3])
    # clustering = KMeans(n_clusters=18, n_init=10).fit(points[:, :3])
    boxArray = xyz_dimension_xyz(points, clustering.labels_, boxArray)
    
    return pc_msg, boxArray


def publish():
    rospy.init_node('me5413_1', anonymous=True)
    rate = rospy.Rate(20) # 20hz
    v = 0
    # print(v)
    pub_boxes = rospy.Publisher('/me5413/bboxes', BoundingBoxArray, queue_size=1)
    pub_cloud = rospy.Publisher('/me5413/pointcloud2', PointCloud2, queue_size=1)
    points_list = [points0, points1, points2, points3, points4, points5, points6, points7, points8, points9]
    while not rospy.is_shutdown():
        v += 1

        points = points_list[1]
        mask = np.ones(points.shape[0], dtype=bool)
        mask = np.logical_and(mask, points[:, 2] > -1.0)
        points = points[mask, :]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd_ds = pcd.voxel_down_sample(voxel_size=2.0)
        points_new = np.asarray(pcd_ds.points)
        points_new = np.hstack((points_new, 50*np.ones((points_new.shape[0], 1))))
        print(points_new.shape)
        # print(v%10)
        pc_msg, boxArray = visualize1cloud(points=points_new, eps=0.6, min_sample=6)
        pub_boxes.publish(boxArray)
        pub_cloud.publish(pc_msg)
        rate.sleep()

if __name__ == '__main__':
    publish()



# num = points1.shape[0]
# print(f"max:{max_label}")
# colors = np.zeros((num, 3))
# color_set = np.zeros((max_label + 1, 3))
# for label in range(max_label + 1):
#     color_set[label, :] = np.random.rand(1, 3)
# for i in range(num):
#     if clustering.labels_[i] == -1:
#         colors[i, :] = np.array([1, 1, 1])
#     else:
#         colors[i, :] = color_set[clustering.labels_[i], :]


# o3d_pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points1))
# colorsp = np.random.rand(points1.shape[0], 3)
# # print(colors)
# o3d_pcd.colors = open3d.utility.Vector3dVector(colors)
# open3d.io.write_point_cloud(os.path.join(root_path, '1.pcd'), o3d_pcd)

