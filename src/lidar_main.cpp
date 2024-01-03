/***************************************************************************************************************************
@file:src/lidar_usv_det.cpp
@author:ljn
@date:2021.12
@brief: detect the usv via lidar sensor
***************************************************************************************************************************/
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
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

// OpenCV
#include "opencv2/opencv.hpp"

// Local
#include "lshape_estimator.hpp"

// Define
#define LEFT_IS_HAP
#define PRINT_LEVEL 1

using namespace std;

// ROS 订阅与接收定义
ros::Publisher postPCPub, objectPub;
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
    // Calculate: 
    // 1. min and max z for cylinder length
    // 2. average x, y, z?
    // 3. fill the cvPoints
    pcl::PointXYZI centroid;
    centroid.x = 0;
    centroid.y = 0;
    centroid.z = 0;
    float zMin = 0;
    float zMax = 0;
    cv::Mat_<float> cvPoints((int)cluster.size(), 2);
    for (size_t i = 0; i < cluster.size(); ++i) {
        centroid.x += cluster.points[i].x;
        centroid.y += cluster.points[i].y;
        centroid.z += cluster.points[i].z;

        cvPoints(i, 0) = cluster.points[i].x;  // x
        cvPoints(i, 1) = cluster.points[i].y;  // y

        if (cluster.points[i].z < zMin || i == 0) {
            zMin = cluster.points[i].z;
        }
            
        if (zMax < cluster.points[i].z || i == 0) {
            zMax = cluster.points[i].z;
            highestX = cluster.points[i].x;
            highestY = cluster.points[i].y;
            highestZ = cluster.points[i].z;
        }
            
    }
    centroid.x = centroid.x / static_cast<float>(cluster.size());
    centroid.y = centroid.y / static_cast<float>(cluster.size());
    centroid.z = centroid.z / static_cast<float>(cluster.size());
   
    // Calculate circumscribed circle on x-y plane
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(cv::Mat(cvPoints).reshape(2), center, radius);

    // Paper : Algo.2 Search-Based Rectangle Fitting
    Eigen::Vector2f e1Star(std::cos(yawEstimate), std::sin(yawEstimate));  // col.11, Algo.2
    Eigen::Vector2f e2Star(-std::sin(yawEstimate), std::cos(yawEstimate));

    std::vector<float> C1Star;  // col.11, Algo.2
    std::vector<float> C2Star;  // col.11, Algo.2
    C1Star.reserve(cluster.size());
    C2Star.reserve(cluster.size());
    for (const auto& point : cluster) {
        C1Star.push_back(point.x * e1Star.x() + point.y * e1Star.y());
        C2Star.push_back(point.x * e2Star.x() + point.y * e2Star.y());
    }

    // col.12, Algo.2
    const float C1StarMin = *std::min_element(C1Star.begin(), C1Star.end());
    const float C1StarMax = *std::max_element(C1Star.begin(), C1Star.end());
    const float C2StarMin = *std::min_element(C2Star.begin(), C2Star.end());
    const float C2StarMax = *std::max_element(C2Star.begin(), C2Star.end());

    const float a1 = std::cos(yawEstimate);
    const float b1 = std::sin(yawEstimate);
    const float c1 = C1StarMin;
    const float a2 = -1.0 * std::sin(yawEstimate);
    const float b2 = std::cos(yawEstimate);
    const float c2 = C2StarMin;
    const float a3 = std::cos(yawEstimate);
    const float b3 = std::sin(yawEstimate);
    const float c3 = C1StarMax;
    const float a4 = -1.0 * std::sin(yawEstimate);
    const float b4 = std::cos(yawEstimate);
    const float c4 = C2StarMax;

    // calc center of bounding box
    float intersection_x_1 = (b1 * c2 - b2 * c1) / (a2 * b1 - a1 * b2);
    float intersection_y_1 = (a1 * c2 - a2 * c1) / (a1 * b2 - a2 * b1);
    float intersection_x_2 = (b3 * c4 - b4 * c3) / (a4 * b3 - a3 * b4);
    float intersection_y_2 = (a3 * c4 - a4 * c3) / (a3 * b4 - a4 * b3);

    // calc dimention of bounding box
    Eigen::Vector2d ex;
    Eigen::Vector2d ey;
    ex << a1 / (std::sqrt(a1 * a1 + b1 * b1)), b1 / (std::sqrt(a1 * a1 + b1 * b1));
    ey << a2 / (std::sqrt(a2 * a2 + b2 * b2)), b2 / (std::sqrt(a2 * a2 + b2 * b2));
    Eigen::Vector2d diagVec;
    diagVec << intersection_x_1 - intersection_x_2, intersection_y_1 - intersection_y_2;

    cenX = (intersection_x_1 + intersection_x_2) / 2.0f;
    cenY = (intersection_y_1 + intersection_y_2) / 2.0f;
    cenZ = centroid.z;
    length = std::fabs(ex.dot(diagVec));
    width = std::fabs(ey.dot(diagVec));

}

void imuCallback(const sensor_msgs::Imu& msg) {
    isIMUSub = true;
    imuPose = Eigen::Quaterniond(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
}

void lidarCallback(const sensor_msgs::PointCloud2::ConstPtr& lidar0, const sensor_msgs::PointCloud2::ConstPtr& lidar1, const sensor_msgs::PointCloud2::ConstPtr& lidar2) {
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
    *lidarRawPC = *lidar1PC + *lidar2PC + *lidar3PC;
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

        // 现在的点应该是 OK 的
        Eigen::Vector3d thisPoint(lidarRawPC->points[i].x, lidarRawPC->points[i].y, lidarRawPC->points[i].z);
        thisPoint = imuPose * thisPoint;

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
    geometry_msgs::PoseArray objects;
    objects.header.stamp = ros::Time::now();
    objects.header.frame_id = "map";
    objects.poses.reserve(clusterIndices.size());
#pragma omp parallel for  
    for (std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin(); it != clusterIndices.end(); ++it) {
        // 创建临时保存点云簇的点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudCluster(new pcl::PointCloud<pcl::PointXYZI>);

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
        orientation_calc orient_calc_("VARIANCE");
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
            ROS_WARN("Cluster %d does not look like a vessel since it has a length = %.2f and width = %.2f.", static_cast<int>(it - clusterIndices.begin() + 1), length, width);
            continue;
        }

        // 如果拟合成功，且长宽符合要求，发布
        geometry_msgs::Pose objPose;
        tf2::Quaternion quat;

        quat.setEuler(0, 0, yawEstimate);

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

    // Publish processed pointcloud
    sensor_msgs::PointCloud2 postPC;
    pcl::toROSMsg(*pc, postPC);
    postPC.header.stamp = ros::Time::now();
    postPC.header.frame_id = "map";
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

void tmplidarCb(const sensor_msgs::PointCloud2::ConstPtr& lidar0) {
    sensor_msgs::PointCloud2 cloud;
    cloud.height = 1;
    cloud.width = 1;
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(4, 
                                 "x", 1, sensor_msgs::PointField::FLOAT32,
                                 "y", 1, sensor_msgs::PointField::FLOAT32,
                                 "z", 1, sensor_msgs::PointField::FLOAT32,
                                 "intensity", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(cloud.width);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloud, "intensity");

    for (size_t i = 0; i < cloud.width; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
        // 假设您有一个数组或类似的结构来存储您的点和强度
        *iter_x = 0;
        *iter_y = 0;
        *iter_z = 0;
        *iter_intensity = 0.517;
    }

    lidarCallback(lidar0, boost::make_shared<const sensor_msgs::PointCloud2>(cloud), boost::make_shared<const sensor_msgs::PointCloud2>(cloud));
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

    ROS_INFO("============ USV Lidar data process program ============");

    // 定义发送
    postPCPub = nh.advertise<sensor_msgs::PointCloud2>("/filter/lidar", 1);
    objectPub = nh.advertise<geometry_msgs::PoseArray>("/filter/target", 1);

    // 定义订阅
    imuSub = nh.subscribe("/usv/imu/data", 1, imuCallback);

    // 定义时间同步订阅
    #ifdef LEFT_IS_HAP
        message_filters::Subscriber<sensor_msgs::PointCloud2> subLidar0(nh, "/livox/lidar_192_168_147_234", 3);
    #else
        message_filters::Subscriber<sensor_msgs::PointCloud2> subLidar0(nh, "/livox/lidar_192_168_147_231", 3);
    #endif
    message_filters::Subscriber<sensor_msgs::PointCloud2> subLidar1(nh, "/livox/lidar_192_168_147_232", 3);
    message_filters::Subscriber<sensor_msgs::PointCloud2> subLidar2(nh, "/livox/lidar_192_168_147_233", 3);

    // 使用 ApproximateTime 策略定义同步器
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(3), subLidar0, subLidar1, subLidar2);
    
    // 注册回调函数，当消息时间接近并准备同步时，该函数会被调用
    sync.registerCallback(boost::bind(&lidarCallback, _1, _2, _3));

    // TMP
    ros::Subscriber lidarSub = nh.subscribe("/hesai/pandar", 1, tmplidarCb);

    // 预分配内存
    lidar1PC->points.reserve(250000);
    lidar2PC->points.reserve(250000);
    lidar3PC->points.reserve(250000);
    lidarRawPC->points.reserve(750000);
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
