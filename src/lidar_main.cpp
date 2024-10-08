// C base
#include <iostream>
#include <vector>
#include <math.h>

// ROS
#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Odometry.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

// Eigen 几何模块
#include <Eigen/Core>
#include <Eigen/Geometry>

// PCL
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

// OpenCV
#include "opencv2/opencv.hpp"

// Local
#include "lshape_estimator.hpp"

// Define
// #define LEFT_IS_HAP
#define PRINT_LEVEL 1

using namespace std;

// ROS 订阅与接收定义
ros::Publisher postPCPub, objectPub, objectBodyPub;
ros::Subscriber imuSub, lidarSub;
float X_MIN = 0.0f;
float X_MAX = 0.0f;
float Y_MIN = 0.0f;
float Y_MAX = 0.0f;
float Z_MIN = 0.0f;
float Z_MAX = 0.0f;
float INTEN_MIN = 0.0f;
float INTEN_MAX = 0.0f;
float USV_LENGTH = 0.0f;
float USV_WIDTH = 0.0f;
float USV_INTENSITY_MIN = 0.0f;
int OUTLIER_STATIC_CHECK_POINT = 0;
float OUTLIER_STATIC_TOL = 0.0f;
float OUTLIER_RADIUS_SEARCH = 0.0f;
int OUTLIER_RADIUS_MIN_NEIGHBOR = 0.0f;
float VOXEL_GRID_DOWNSAMPLE_FACTOR = 0.0f;
int CLUSTER_SIZE_MIN = 0;
int CLUSTER_SIZE_MAX = 0;
float CLUSTER_TOL = 0.0f;
float TARGET_VESSEL_LENGTH_MIN = 0.0f;
float TARGET_VESSEL_LENGTH_MAX = 0.0f;
float TARGET_VESSEL_WIDTH_MIN = 0.0f;
float TARGET_VESSEL_WIDTH_MAX = 0.0f;

// 处理计数器
int counter = 0;

// 点云定义
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar1PC(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar2PC(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar3PC(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar4PC(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar5PC(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr lidarRawPC(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
std::vector<pcl::PointIndices> clusterIndices;

// 创建离群点滤波器对象
pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
pcl::RadiusOutlierRemoval<pcl::PointXYZI> ror;

// 定义 VoxelGrid 滤波器变量
pcl::VoxelGrid<pcl::PointXYZI> voxGrid;

// 点云欧式聚类分割，基于 KdTree 对象作为搜索方法
pcl::search::KdTree<pcl::PointXYZI>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZI>);
pcl::EuclideanClusterExtraction<pcl::PointXYZI> ecExtraction;

// IMU 定义
bool isIMUSub = false;
Eigen::Quaterniond imuPose;

// 方向估计
OrientationCalc orient_calc_("VARIANCE");

// 点云过滤离群点，叶素滤波
inline void cloudFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr) {

    // 判断点云是否为空
    if (cloudPtr->size() == 0) {
        return;
    }

    // 去除离群点
    // sor.setInputCloud(cloudPtr);
    // sor.filter(*cloudPtr);
    ror.setInputCloud(cloudPtr);
    ror.filter(*cloudPtr);

    // 叶素滤波
    voxGrid.setInputCloud(cloudPtr);
    voxGrid.filter(*cloudPtr);

    return;
}

inline void clusterExt(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr) {
    clusterIndices.clear();

    // 判断点云是否为空
    if (cloudPtr->size() == 0) {
        return;
    }
    
    // 点云输入 KdTree 对象
    kdTree->setInputCloud(cloudPtr);
 
    // 设置被搜索的点云
    ecExtraction.setInputCloud(cloudPtr);     

    // 聚类抽取结果保存在一个数组中，数组中每个元素代表抽取的一个组件点云的下标
    ecExtraction.extract(clusterIndices);

    return;
}

inline void clusterDirection(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr, float objVector[3], float& len, float& wid, float& lwRatio, float& cenX, float& cenY) {
    // 判断点云是否为空
    if (cloudPtr->size() == 0) {
        return;
    }
    
    pcl::MomentOfInertiaEstimation<pcl::PointXYZI> feature_extractor;
    feature_extractor.setInputCloud(cloudPtr);
    feature_extractor.compute();

    std::vector<float> moment_of_inertia;
    std::vector<float> eccentricity;
    pcl::PointXYZI min_point_OBB;
    pcl::PointXYZI max_point_OBB;
    pcl::PointXYZI position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;
    Eigen::Vector3f mass_center;

    feature_extractor.getMomentOfInertia(moment_of_inertia);                                      // 计算出的惯性矩
    feature_extractor.getEccentricity(eccentricity);                                              // 计算出的偏心率
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);  // OBB对应的相关参数
    feature_extractor.getEigenValues(major_value, middle_value, minor_value);                     // 三个特征值
    feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);                 // 三个特征向量
    feature_extractor.getMassCenter(mass_center);                                                 // 计算质心

    for (int i = 0; i < 3; i++) {
        objVector[i] = major_vector[i];
    }
    
    len = max_point_OBB.x - min_point_OBB.x;
    wid = max_point_OBB.y - min_point_OBB.y;
    lwRatio = len/wid;
    cenX = mass_center[0];
    cenY = mass_center[1];

    return;
}

inline void calculateDimPos(const pcl::PointCloud<pcl::PointXYZI>& cluster, float yawEstimate,  float& cenX, float& cenY, float& cenZ, float& length, float& width, float& highestX, float& highestY, float& highestZ) {

    // Calculate the center of the cluster
    // Transform x and y according to yawEstimate, record min and max after the transform
    std::vector<float> C1Star(cluster.size());
    std::vector<float> C2Star(cluster.size());
    float C1StarMin = cluster.points[0].x * std::cos(yawEstimate) + cluster.points[0].y * std::sin(yawEstimate);
    float C1StarMax = cluster.points[0].x * std::cos(yawEstimate) + cluster.points[0].y * std::sin(yawEstimate);
    float C2StarMin = cluster.points[0].x * (-std::sin(yawEstimate)) + cluster.points[0].y * std::cos(yawEstimate);
    float C2StarMax = cluster.points[0].x * (-std::sin(yawEstimate)) + cluster.points[0].y * std::cos(yawEstimate);
    Eigen::Vector3f sumXYZ(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f centerXYZ(0.0f, 0.0f, 0.0f);
    for (size_t i = 0; i < cluster.size(); ++i) {
        sumXYZ[0] += cluster.points[i].x;
        sumXYZ[1] += cluster.points[i].y;
        sumXYZ[2] += cluster.points[i].z;
        C1Star[i] = cluster.points[i].x * std::cos(yawEstimate) + cluster.points[i].y * std::sin(yawEstimate);
        C2Star[i] = cluster.points[i].x * (-std::sin(yawEstimate)) + cluster.points[i].y * std::cos(yawEstimate);
        if (C1Star[i] > C1StarMax) {
            C1StarMax = C1Star[i];
        }
        if (C1Star[i] < C1StarMin) {
            C1StarMin = C1Star[i];
        }
        if (C2Star[i] > C2StarMax) {
            C2StarMax = C2Star[i];
        }
        if (C2Star[i] < C2StarMin) {
            C2StarMin = C2Star[i];
        }
    }
    centerXYZ = sumXYZ / static_cast<float>(cluster.size());

    // Sort the cluster by z
    std::vector<pcl::PointXYZI> sortedCluster(cluster.points.begin(), cluster.points.end());
    std::sort(sortedCluster.begin(), sortedCluster.end(), [](const pcl::PointXYZI& a, const pcl::PointXYZI& b) {
        return a.z > b.z;
    });

    // Calculate the center of the 10% highest point
    size_t top_count = static_cast<size_t>(std::ceil(sortedCluster.size() * 0.1f));
    Eigen::Vector3f partSumXYZ(0.0f, 0.0f, 0.0f);
    Eigen::Vector3f highestPartCenterXYZ(0.0f, 0.0f, 0.0f);
    for (size_t i = 0; i < top_count; ++i) {
        partSumXYZ[0] += sortedCluster[i].x;
        partSumXYZ[1] += sortedCluster[i].y;
        partSumXYZ[2] += sortedCluster[i].z;
    }
    highestPartCenterXYZ = partSumXYZ / static_cast<float>(top_count);

    // Calculate circumscribed circle on x-y plane using Search-Based Rectangle Fitting
    const float a1 = std::cos(yawEstimate);
    const float b1 = std::sin(yawEstimate);
    const float c1 = C1StarMin;
    const float a2 = -1.0f * std::sin(yawEstimate);
    const float b2 = std::cos(yawEstimate);
    const float c2 = C2StarMin;
    const float a3 = std::cos(yawEstimate);
    const float b3 = std::sin(yawEstimate);
    const float c3 = C1StarMax;
    const float a4 = -1.0f * std::sin(yawEstimate);
    const float b4 = std::cos(yawEstimate);
    const float c4 = C2StarMax;

    // Calculate the center of the bounding box
    float intersectionX1 = (b1 * c2 - b2 * c1) / (a2 * b1 - a1 * b2);
    float intersectionY1 = (a1 * c2 - a2 * c1) / (a1 * b2 - a2 * b1);
    float intersectionX2 = (b3 * c4 - b4 * c3) / (a4 * b3 - a3 * b4);
    float intersectionY2 = (a3 * c4 - a4 * c3) / (a3 * b4 - a4 * b3);

    // Calculate the dimension of the bounding box
    Eigen::Vector2f ex(a1 / (std::sqrt(a1 * a1 + b1 * b1)), b1 / (std::sqrt(a1 * a1 + b1 * b1)));
    Eigen::Vector2f ey(a2 / (std::sqrt(a2 * a2 + b2 * b2)), b2 / (std::sqrt(a2 * a2 + b2 * b2)));
    Eigen::Vector2f diagVec(intersectionX1 - intersectionX2, intersectionY1 - intersectionY2);

    cenX = (intersectionX1 + intersectionX2) / 2.0f;
    cenY = (intersectionY1 + intersectionY2) / 2.0f;
    cenZ = centerXYZ.z();
    length = std::fabs(ex.dot(diagVec));
    width = std::fabs(ey.dot(diagVec));
    highestX = highestPartCenterXYZ[0];
    highestY = highestPartCenterXYZ[1];
    highestZ = highestPartCenterXYZ[2];
}

void imuCallback(const sensor_msgs::Imu& msg) {
    isIMUSub = true;
    imuPose = Eigen::Quaterniond(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
}

void lidarCallback(const sensor_msgs::PointCloud2::ConstPtr& lidar0, const sensor_msgs::PointCloud2::ConstPtr& lidar1, const sensor_msgs::PointCloud2::ConstPtr& lidar2, const sensor_msgs::PointCloud2::ConstPtr& lidar3, const sensor_msgs::PointCloud2::ConstPtr& lidar4) {
    // Record init time    
    #ifdef DEBUG
        #if PRINT_LEVEL > 0
            int boxedPointNum = 0;
            int inlierPointNum = 0;
            #if PRINT_LEVEL > 1
                ros::Time t0 = ros::Time::now();
                ros::Time t1 = ros::Time::now();
                ros::Time t2 = ros::Time::now();
                ros::Time t3 = ros::Time::now();
                ros::Time t4 = ros::Time::now();
                ros::Time t5 = ros::Time::now();
            #endif
        #endif
    #endif

    // Merge 3 pointclouds
    pcl::fromROSMsg(*lidar0, *lidar1PC);
    pcl::fromROSMsg(*lidar1, *lidar2PC);
    pcl::fromROSMsg(*lidar2, *lidar3PC);
    pcl::fromROSMsg(*lidar3, *lidar4PC);
    pcl::fromROSMsg(*lidar4, *lidar5PC);
    *lidarRawPC = *lidar1PC + *lidar2PC + *lidar3PC + *lidar4PC + *lidar5PC;
    #ifdef DEBUG
        #if PRINT_LEVEL > 0
            #if PRINT_LEVEL > 1
                t1 = ros::Time::now();
            #endif
        #endif
    #endif
 
    // Cut filter
#pragma omp parallel for
    for (size_t i = 0; i < lidarRawPC->size(); i++) {
        // Intensity
        if(lidarRawPC->points[i].intensity < INTEN_MIN) {
            continue;
        }

        // Outer box filter
        if (lidarRawPC->points[i].x < X_MIN || lidarRawPC->points[i].x > X_MAX) {
            continue;
        }
        if (lidarRawPC->points[i].y < Y_MIN || lidarRawPC->points[i].y > Y_MAX) {
            continue;
        }
        if (lidarRawPC->points[i].z < Z_MIN || lidarRawPC->points[i].z > Z_MAX) {
            continue;
        }

        // USV filter
        if (lidarRawPC->points[i].x < USV_LENGTH && lidarRawPC->points[i].x > -USV_LENGTH && lidarRawPC->points[i].y < USV_WIDTH && lidarRawPC->points[i].y > -USV_WIDTH) {
            continue;
        }

        // hook filter
        if (1.80f < lidarRawPC->points[i].y && lidarRawPC->points[i].y < 2.50f && 0.1f < lidarRawPC->points[i].z && lidarRawPC->points[i].z <= 0.9f) {
            if ((0.1f < lidarRawPC->points[i].x && lidarRawPC->points[i].x < 0.8f) || 
            (-0.9f < lidarRawPC->points[i].x && lidarRawPC->points[i].x < -0.53f) || 
            (-2.66f < lidarRawPC->points[i].x && lidarRawPC->points[i].x < -2.36f)) {
                continue;
            }

        }
        // if ((0.1f < lidarRawPC->points[i].x && lidarRawPC->points[i].x < 0.8f) || 
        //     (-0.9f < lidarRawPC->points[i].x && lidarRawPC->points[i].x < -0.53f) || 
        //     (-2.66f < lidarRawPC->points[i].x && lidarRawPC->points[i].x < -2.36f)) {
        //     if ((1.8f < lidarRawPC->points[i].y && lidarRawPC->points[i].y < 2.05f) && 
        //         (0.1f < lidarRawPC->points[i].z && lidarRawPC->points[i].z <= 0.61f)) {
        //         continue;
        //     }
        //     if ((1.97f < lidarRawPC->points[i].y && lidarRawPC->points[i].y < 2.43f) && 
        //         (0.4f < lidarRawPC->points[i].z && lidarRawPC->points[i].z < 0.90f)) {
        //         continue;
        //     }
        // }

        // 现在的点应该是 OK 的
        Eigen::Vector3d thisPoint(lidarRawPC->points[i].x, lidarRawPC->points[i].y, lidarRawPC->points[i].z);
        thisPoint = imuPose * thisPoint; // Body -> ENU

        pcl::PointXYZI point_pcl = lidarRawPC->points[i];
        point_pcl.x = thisPoint[0];
        point_pcl.y = thisPoint[1];
        point_pcl.z = thisPoint[2];
        pc->push_back(point_pcl);
    }
    #ifdef DEBUG
        #if PRINT_LEVEL > 0
            boxedPointNum = pc->size();
            #if PRINT_LEVEL > 1
                t2 = ros::Time::now();
            #endif
        #endif
    #endif
    
    // Outlier filters
    cloudFilter(pc);
    #ifdef DEBUG
        #if PRINT_LEVEL > 0
            inlierPointNum = pc->size();
            #if PRINT_LEVEL > 1
                t3 = ros::Time::now();
            #endif
        #endif
    #endif

    // Cluster extraction
    clusterExt(pc);
    #ifdef DEBUG
        #if PRINT_LEVEL > 0
            #if PRINT_LEVEL > 1
                t4 = ros::Time::now();
            #endif
        #endif
    #endif

    // Cluster state estimation
    geometry_msgs::PoseArray objects, objects_b;
    objects.header.stamp = ros::Time::now();
    objects.header.frame_id = "map";
    objects.poses.reserve(2 * clusterIndices.size() + 5);
    objects_b.header.stamp = ros::Time::now();
    objects_b.header.frame_id = "base_link";
    objects_b.poses.reserve(clusterIndices.size() + 5);
#pragma omp parallel for  
    for (std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin(); it != clusterIndices.end(); ++it) {
        // 创建临时保存点云簇的点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudCluster(new pcl::PointCloud<pcl::PointXYZI>);
        cloudCluster->points.reserve(it->indices.size() + 100);

        // 通过下标，逐个填充
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++) {
            cloudCluster->points.push_back(pc->points[*pit]);
        }

        // 提取点云位姿
        float yawEstimate = 0.0f;
        float length = 0.0f;
        float width = 0.0f;
        float cenX = 0.0f;
        float cenY = 0.0f;
        float cenZ = 0.0f;
        float highestX = 0.0f;
        float highestY = 0.0f;
        float highestZ = 0.0f;
        bool isSuccessFitted = orient_calc_.LshapeFitting(*cloudCluster, yawEstimate);

        // 如果拟合失败，跳过
        if (!isSuccessFitted) {
            ROS_WARN("L-shape fitting failed in Cluster %d.", static_cast<int>(it - clusterIndices.begin() + 1));
            continue;
        }

        calculateDimPos(*cloudCluster, yawEstimate, cenX, cenY, cenZ, length, width, highestX, highestY, highestZ);
        if (length < width) {
            // 交换长宽
            std::swap(length, width);

            // 估计的航向增加 90 度，并且将其限制在 [-pi/2, pi/2] 之间
            yawEstimate = yawEstimate + 0.5 * M_PI;
            yawEstimate = atan(tan(yawEstimate));
        }

        // 如果拟合成功，但是长宽不符合要求，跳过
        if (length > TARGET_VESSEL_LENGTH_MAX || width > TARGET_VESSEL_WIDTH_MAX || length < TARGET_VESSEL_LENGTH_MIN || width < TARGET_VESSEL_WIDTH_MIN) {
            ROS_WARN("Cluster %d/%d does not look like a vessel since it has a length = %.2f and width = %.2f.", static_cast<int>(it - clusterIndices.begin() + 1), static_cast<int>(clusterIndices.end() - clusterIndices.begin() + 1), length, width);
            continue;
        }

        // 如果拟合成功，且长宽符合要求，发布
        geometry_msgs::Pose objPose;
        // tf2::Quaternion quat;
        Eigen::AngleAxisd rollAngle(0, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(0, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yawEstimate, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond quat = yawAngle * pitchAngle * rollAngle;

        // quat.setEuler(0, 0, yawEstimate);

        objPose.position.x = cenX;
        objPose.position.y = cenY;
        objPose.position.z = cenZ;
        objPose.orientation.w = quat.w();
        objPose.orientation.x = quat.x();
        objPose.orientation.y = quat.y();
        objPose.orientation.z = quat.z();
        objects.poses.push_back(objPose);

        objPose.position.x = highestX;
        objPose.position.y = highestY;
        objPose.position.z = highestZ;
        objPose.orientation.w = 0;
        objPose.orientation.x = length;
        objPose.orientation.y = width;
        objPose.orientation.z = 0;
        objects.poses.push_back(objPose);

        Eigen::Vector3d bodyPoint(cenX, cenY, cenZ);
        bodyPoint = imuPose.conjugate() * bodyPoint;
        quat = imuPose.conjugate() * quat;
        objPose.position.x = bodyPoint(0);
        objPose.position.y = bodyPoint(1);
        objPose.position.z = bodyPoint(2);
        objPose.orientation.w = quat.w();
        objPose.orientation.x = quat.x();
        objPose.orientation.y = quat.y();
        objPose.orientation.z = quat.z();
        objects_b.poses.push_back(objPose);
    }
    #ifdef DEBUG
        #if PRINT_LEVEL > 0
            #if PRINT_LEVEL > 1
                t5 = ros::Time::now();
            #endif
        #endif
    #endif
    
    // Publish object 
    objectPub.publish(objects); 
    objectBodyPub.publish(objects_b);

    // Publish processed pointcloud
#pragma omp parallel for
    for (size_t i = 0; i < pc->size(); i++) {
        Eigen::Vector3d thisPoint(pc->points[i].x, pc->points[i].y, pc->points[i].z);
        thisPoint = imuPose.conjugate() * thisPoint;
        pc->points[i].x = thisPoint[0];
        pc->points[i].y = thisPoint[1];
        pc->points[i].z = thisPoint[2];
    }
    sensor_msgs::PointCloud2 postPC;
    pcl::toROSMsg(*pc, postPC);
    postPC.header.stamp = ros::Time::now();
    postPC.header.frame_id = "base_link";
    postPCPub.publish(postPC);
    pc->clear();

    // Print debug info
    #ifdef DEBUG
        #if PRINT_LEVEL > 0
            ROS_INFO("=============== No. %d ===============", counter++);
            ROS_INFO("Point number in raw pointcloud: %d", lidarRawPC->size());
            ROS_INFO("Point number in boxed pointcloud: %d", boxedPointNum); 
            ROS_INFO("Point number in outlier-removed pointcloud: %d", inlierPointNum);
            ROS_INFO("Cluster number: %d", static_cast<int>(clusterIndices.size()));

            #if PRINT_LEVEL > 1
                ROS_INFO("Merge all point clouds costs: %e s", (t1 - t0).tosec()); 
                ROS_INFO("Box filter costs: %e s", (t2 - t1).tosec());
                ROS_INFO("Removing outliers costs: %e s", (t3 - t2).tosec());
                ROS_INFO("Cluster extraction costs: %e s", (t4 - t3).tosec());
                ROS_INFO("State estimation finished costs: %e s", (t5 - t4).tosec());
                ROS_INFO("This callback costs: %e s\n", (t5 - t0).tosec());
            #endif 
        #endif
    #endif
}

int main(int argc, char** argv) {
    // 初始化 ROS 节点
    ros::init(argc, argv, "lidar_usv_det");
    ros::NodeHandle nh("~");
    ros::Rate rate(10.0);

    // 加载参数
    nh.param<float>("x_min", X_MIN, -200.0f);
    nh.param<float>("x_max", X_MAX, 200.0f);
    nh.param<float>("y_min", Y_MIN, -70.0f);
    nh.param<float>("y_max", Y_MAX, 70.0f);
    nh.param<float>("z_min", Z_MIN, -2.0f);
    nh.param<float>("z_max", Z_MAX, 4.0f);
    nh.param<float>("inten_min", INTEN_MIN, 20.0f);
    nh.param<float>("inten_max", INTEN_MAX, 100.0f);
    nh.param<float>("usv_length", USV_LENGTH, 3.6f);
    nh.param<float>("usv_width", USV_WIDTH, 1.25f);
    nh.param<float>("usv_intensity", USV_INTENSITY_MIN, 20);
    nh.param<int>("outlier_static_check_point", OUTLIER_STATIC_CHECK_POINT, 50);
    nh.param<float>("outlier_static_tol", OUTLIER_STATIC_TOL, 1.0f);
    nh.param<float>("outlier_radius_search", OUTLIER_RADIUS_SEARCH, 6.0f);
    nh.param<int>("outlier_radius_min_neighbor", OUTLIER_RADIUS_MIN_NEIGHBOR, 4);
    nh.param<float>("voxel_grid_downsample_factor", VOXEL_GRID_DOWNSAMPLE_FACTOR, 0.2f);
    nh.param<int>("cluster_size_min", CLUSTER_SIZE_MIN, 30);
    nh.param<int>("cluster_size_max", CLUSTER_SIZE_MAX, 10000);
    nh.param<float>("cluster_tol", CLUSTER_TOL, 3.0f);
    nh.param<float>("target_vessel_length_min", TARGET_VESSEL_LENGTH_MIN, 1.0f);
    nh.param<float>("target_vessel_length_max", TARGET_VESSEL_LENGTH_MAX, 25.0f);
    nh.param<float>("target_vessel_width_min", TARGET_VESSEL_WIDTH_MIN, 0.25f);
    nh.param<float>("target_vessel_width_max", TARGET_VESSEL_WIDTH_MAX, 8.0f);

    
    ROS_INFO("============ USV Lidar data process program STARTED! ============");

    // 定义发送
    postPCPub = nh.advertise<sensor_msgs::PointCloud2>("/filter/lidar", 1);
    objectPub = nh.advertise<geometry_msgs::PoseArray>("/filter/target", 1);
    objectBodyPub = nh.advertise<geometry_msgs::PoseArray>("/filter/target_b", 1);

    // 定义订阅
    imuSub = nh.subscribe("/usv/imu/data", 1, imuCallback);

    // 定义时间同步订阅
    message_filters::Subscriber<sensor_msgs::PointCloud2> subLidar0(nh, "/livox/lidar_192_168_147_231", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> subLidar1(nh, "/livox/lidar_192_168_147_232", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> subLidar2(nh, "/livox/lidar_192_168_147_233", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> subLidar3(nh, "/livox/lidar_192_168_147_234", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> subLidar4(nh, "/livox/lidar_192_168_147_230", 10);

    // 使用 ApproximateTime 策略定义同步器
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(4), subLidar0, subLidar1, subLidar2, subLidar3, subLidar4);
    
    // 注册回调函数，当消息时间接近并准备同步时，该函数会被调用
    sync.registerCallback(boost::bind(&lidarCallback, _1, _2, _3, _4, _5));

    // 预分配内存
    lidar1PC->points.reserve(250000);
    lidar2PC->points.reserve(250000);
    lidar3PC->points.reserve(250000);
    lidar4PC->points.reserve(250000);
    lidar5PC->points.reserve(250000);
    lidarRawPC->points.reserve(1000000);
    pc->points.reserve(500000);
    clusterIndices.reserve(20);

    // 点云滤波器处理设置
    
    // 离群值滤波器设置
    sor.setMeanK(OUTLIER_STATIC_CHECK_POINT);       // 设置在进行统计时考虑查询点邻近点数
    sor.setStddevMulThresh(OUTLIER_STATIC_TOL);     // 设置判断是否为离群点的阈值
    ror.setRadiusSearch(OUTLIER_RADIUS_SEARCH);     
    ror.setMinNeighborsInRadius(OUTLIER_RADIUS_MIN_NEIGHBOR);

    // 叶素滤波器设置
    voxGrid.setLeafSize(VOXEL_GRID_DOWNSAMPLE_FACTOR, VOXEL_GRID_DOWNSAMPLE_FACTOR, VOXEL_GRID_DOWNSAMPLE_FACTOR);

    // 聚类分割设置
    ecExtraction.setClusterTolerance(CLUSTER_TOL);  // 点之间相隔大于10m的不被认为是一个聚类
    ecExtraction.setMinClusterSize(CLUSTER_SIZE_MIN);   // 一个聚类里所允许的最少点数
    ecExtraction.setMaxClusterSize(CLUSTER_SIZE_MAX);   // 一个聚类里所允许的最多点数
    ecExtraction.setSearchMethod(kdTree);   // 设置搜索方法为 KdTreee

    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
};
