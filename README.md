## Intro
This ROS package reads pointcloud data from 5 lidars and estimates the position and some attitude information of the clusters.

## Input 
All inputs are in type `sensor_msgs/PointCloud2`.
- `/livox/lidar_192_168_147_230`
- `/livox/lidar_192_168_147_231`
- `/livox/lidar_192_168_147_232`
- `/livox/lidar_192_168_147_233`
- `/livox/lidar_192_168_147_234`

## Output
Outputs are in topics:
- `/filter/lidar` in `sensor_msgs/PointCloud2`: This is the processed pointcloud.
- `/filter/target` in `geometry_msgs/PoseArray`: This is the position and some attitude information of the clusters. Each cluster will occupy two `geometry_msgs/Pose` poses in the topic.
- Therefore the length of messages in the topic = 2 * the number of clusters.
- For the nth (n = 1, 2, ..., n, ...) cluster, its information will be structured as:
    - geometry_msgs/Pose[2*n-2]:
        - position (related to the mass center of the USV)
            - x: x of the cluster center
            - y: y of the cluster center
            - z: z of the cluster center
        - orientation: x, y, z, w are the quaternions of the clusters, only containing the heading (yaw) info of the cluster.
    - geometry_msgs/Pose[2*n-1]:
        - position (related to the mass center of the USV)
            - x: x of the highest point of the cluster
            - y: y of the highest point of the cluster
            - z: z of the highest point of the cluster
        - orientation:
            - x: length of the cluster
            - y: width of the cluster
            - z & w are reserved

## Highlight
- Box filter is used for removing useless or meaningless points
- Outlier filter is used for removing noise points
- L-shape fitting is used for attitude estimation
- Search-based rectangle fitting is used to estimate position
- OpenMP is used for increasing process speed
- Parameters can be adjusted in launch files

 ## Thanks
 L-shape fitting from [L-shape-fitting-3D-LiDAR-ROS](https://github.com/HMX2013/L-shape-fitting-3D-LiDAR-ROS) is used and modified for faster process speed.
