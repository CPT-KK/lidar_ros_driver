/***************************************************************************************************************************
@file:src/lidar_usv_det.cpp
@author:ljn
@date:2021.12
@brief: detect the usv via lidar sensor
***************************************************************************************************************************/
// ROS
#include <geometry_msgs/PoseArray.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <Eigen/Core>

#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Int32MultiArray.h"

// Eigen 几何模块
#include <Eigen/Geometry>
// pcl
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>

// clustering
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/moment_of_inertia_estimation.h>

// opencv
#include "opencv2/opencv.hpp"

// local
#include "lshape_estimator.hpp"

#define USE_IMU
// #define EST_HEADING

using namespace std;

ros::Publisher postPCPub, objectPub;
ros::Subscriber imuSub;
float X_MIN = 0.0f;
float X_MAX = 0.0f;
float Y_MIN = 0.0f;
float Y_MAX = 0.0f;
float Z_MIN = 0.0f;
float Z_MAX = 0.0f;
float USV_LENGTH = 0.0f;
float USV_WIDTH = 0.0f;
float USV_INTENSITY_MIN = 0.0f;
int OUTLIER_STATIC_CHECK_POINT = 0;
float OUTLIER_STATIC_TOL = 0;
int CLUSTER_SIZE_MIN = 0;
int CLUSTER_SIZE_MAX = 0;
float CLUSTER_TOL = 0;
float TARGET_VESSEL_LENGTH_MIN = 0.0f;
float TARGET_VESSEL_LENGTH_MAX = 0.0f;
float TARGET_VESSEL_WIDTH_MIN = 0.0f;
float TARGET_VESSEL_WIDTH_MAX = 0.0f;

int counter = 0;

// 包含功能1：remove the usv body from the lidar pointscloud
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar1PC(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar2PC(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr lidar3PC(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr lidarRawPC(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
std::vector<pcl::PointIndices> clusterIndices;

#ifdef USE_IMU
bool label_imu_sub = false;
Eigen::Quaterniond imu_pose;

inline void cloudFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float divFilter) {

    // 判断点云是否为空
    if (cloud->size() == 0) {
        return;
    }
    
    // 创建离群点滤波器对象
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;

    // 定义 VoxelGrid 滤波器变量
    pcl::VoxelGrid<pcl::PointXYZI> voxGrid;

    // 去除离群点
    sor.setInputCloud(cloud);

    // 设置在进行统计时考虑查询点邻近点数
    sor.setMeanK(100);

    // 设置判断是否为离群点的阈值
    sor.setStddevMulThresh(1.0f);

    // 执行滤波处理保存内点到inputCloud
    sor.filter(*cloud);

    // 叶素滤波
    voxGrid.setInputCloud(cloud);
    voxGrid.setLeafSize(divFilter, divFilter, divFilter);
    voxGrid.filter(*cloud);

    return;
}

inline void clusterExt(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    clusterIndices.clear();

    // 判断点云是否为空
    if (cloud->size() == 0) {
        return;
    }
    
    // 点云欧式聚类分割，基于 KdTree 对象作为搜索方法
    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZI>);
    kdTree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ecExtraction;
    ecExtraction.setClusterTolerance(CLUSTER_TOL);      // 点之间相隔大于10m的不被认为是一个聚类
    ecExtraction.setMinClusterSize(CLUSTER_SIZE_MIN);       // 一个聚类里所允许的最少点数
    ecExtraction.setMaxClusterSize(CLUSTER_SIZE_MAX);    // 一个聚类里所允许的最多点数
    ecExtraction.setSearchMethod(kdTree);     // 设置搜索方法为 KdTreee
    ecExtraction.setInputCloud(cloud);     // 设置被搜索的点云

    // 聚类抽取结果保存在一个数组中，数组中每个元素代表抽取的一个组件点云的下标
    ecExtraction.extract(clusterIndices);

    return;
}

inline void clusterDirection(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float objVector[3], float& len, float& wid, float& lwRatio, float& cenX, float& cenY) {
    // 判断点云是否为空
    if (cloud->size() == 0) {
        return;
    }
    
    pcl::MomentOfInertiaEstimation<pcl::PointXYZI> feature_extractor;
    feature_extractor.setInputCloud(cloud);
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

    // printf("Center: [%.2f, %.2f, %.2f] | Direction: [%.2f, %.2f, %.2f]\n", mass_center[0], mass_center[1], mass_center[2], major_vector[0], major_vector[1], major_vector[2]);

    return;
}

void imuCallback(const sensor_msgs::Imu& msg) {
    label_imu_sub = true;
    imu_pose = Eigen::Quaterniond(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z);
}
#endif

void lidarcallback(const sensor_msgs::PointCloud2::ConstPtr& lidar0, const sensor_msgs::PointCloud2::ConstPtr& lidar1, const sensor_msgs::PointCloud2::ConstPtr& lidar2) {
    // Record init time
    ros::Time t0 = ros::Time::now();
    ROS_INFO("Processing No.%d", counter++);

    // Merge 3 pointclouds
    pcl::fromROSMsg(*lidar0, *lidar1PC);
    pcl::fromROSMsg(*lidar1, *lidar2PC);
    pcl::fromROSMsg(*lidar2, *lidar3PC);
    *lidarRawPC = *lidar1PC + *lidar2PC + *lidar3PC;
    ROS_INFO("Raw pointcloud: %d", static_cast<int>(lidarRawPC->size())); 
    ROS_INFO("Processed from msgs: %e s", (ros::Time::now() - t0).toSec()); 
    t0 = ros::Time::now();

    // Cut filter
#pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < lidarRawPC->size(); i++) {
        // Intensity
        if(lidarRawPC->points[i].intensity < 10.0f) {
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
        lidarRawPC->points[i].z = 0.0f;
        Eigen::Vector3d thisPoint(lidarRawPC->points[i].x, lidarRawPC->points[i].y, lidarRawPC->points[i].z);
        // thisPoint = imu_pose * thisPoint;

        pcl::PointXYZI point_pcl = lidarRawPC->points[i];
        point_pcl.x = thisPoint[0];
        point_pcl.y = thisPoint[1];
        point_pcl.z = thisPoint[2];
        pc->push_back(point_pcl);
    }
    ROS_INFO("Boxed pointcloud: %d", static_cast<int>(pc->size()));
    ROS_INFO("Box filter costs: %e s", (ros::Time::now() - t0).toSec());
    t0 = ros::Time::now();

    // Outlier filters
    cloudFilter(pc, 0.1f);
    ROS_INFO("Outlier removed pointcloud: %d", static_cast<int>(pc->size()));
    ROS_INFO("Removing outliers costs: %e s", (ros::Time::now() - t0).toSec());
    t0 = ros::Time::now();

    // Cluster extraction
    clusterExt(pc);
    ROS_INFO("Extract %d clusters, cost %e s", static_cast<int>(clusterIndices.size()), (ros::Time::now() - t0).toSec());
    t0 = ros::Time::now();

    // Cluster state estimation
    geometry_msgs::PoseArray objects;
    objects.header.stamp = ros::Time::now();
    objects.header.frame_id = "base_link";
    objects.poses.reserve(clusterIndices.size());
#pragma omp parallel for num_threads(8)    
    for (std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin(); it != clusterIndices.end(); ++it) {
        // 创建临时保存点云簇的点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudCluster(new pcl::PointCloud<pcl::PointXYZI>);

        // 通过下标，逐个填充
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++) {
            cloudCluster->points.push_back(pc->points[*pit]);
        }

        // 提取点云位姿
        geometry_msgs::Pose objPose;
        float objVector[3] = {0.0f, 0.0f, 0.0f};
        float len = 0.0f;
        float wid = 0.0f;
        float lwRatio = 0.0f;
        float cenX = 0.0f;
        float cenY = 0.0f;
        clusterDirection(cloudCluster, objVector, len, wid, lwRatio, cenX, cenY);

        if (len > TARGET_VESSEL_LENGTH_MAX || wid > TARGET_VESSEL_WIDTH_MAX) {
            continue;
        }

        objPose.position.x = cenX;
        objPose.position.y = cenY;
        tf2::Quaternion quat;
        quat.setEuler(0, 0, std::atan(objVector[1]/objVector[0]));

        objPose.orientation.w = quat.w();
        objPose.orientation.x = quat.x();
        objPose.orientation.y = quat.y();
        objPose.orientation.z = quat.z();
        objects.poses.push_back(objPose);
    }
    ROS_INFO("Cluster state estimation finished in %e s", (ros::Time::now() - t0).toSec());
    t0 = ros::Time::now();

    // Publish object 
    objectPub.publish(objects); 

    // Publish processed pointcloud
    sensor_msgs::PointCloud2 postPC;
    pcl::toROSMsg(*pc, postPC);
    postPC.header.stamp = lidar0->header.stamp;
    postPC.header.frame_id = "base_link";
    postPCPub.publish(postPC);
    pc->clear();
    ROS_INFO("Processed pointcloud published in %e s\n", (ros::Time::now() - t0).toSec());
}

int main(int argc, char** argv) {
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
    nh.param<float>("usv_length", USV_LENGTH, 3.6f);
    nh.param<float>("usv_width", USV_WIDTH, 1.25f);
    nh.param<float>("usv_intensity", USV_INTENSITY_MIN, 20);
    nh.param<int>("outlier_static_check_point", OUTLIER_STATIC_CHECK_POINT, 50);
    nh.param<float>("outlier_static_tol", OUTLIER_STATIC_TOL, 1.5f);
    nh.param<int>("cluster_size_min", CLUSTER_SIZE_MIN, 50);
    nh.param<int>("cluster_size_max", CLUSTER_SIZE_MAX, 10000);
    nh.param<float>("cluster_tol", CLUSTER_TOL, 20);
    nh.param<float>("target_vessel_length_min", TARGET_VESSEL_LENGTH_MIN, 1.5f);
    nh.param<float>("target_vessel_length_max", TARGET_VESSEL_LENGTH_MAX, 25.0f);
    nh.param<float>("target_vessel_width_min", TARGET_VESSEL_WIDTH_MIN, 0.5f);
    nh.param<float>("target_vessel_width_max", TARGET_VESSEL_WIDTH_MAX, 8.0f);

    ROS_INFO("Params: Intensity: %.2f | X box filter: %.2f | Y box filter: %.2f", USV_INTENSITY_MIN, USV_LENGTH, USV_WIDTH);

    // 定义发送
    postPCPub = nh.advertise<sensor_msgs::PointCloud2>("/filter/lidar", 1);
    objectPub = nh.advertise<geometry_msgs::PoseArray>("/filter/target", 1);

    // 定义订阅
    imuSub = nh.subscribe("/mavros/imu/data", 1, imuCallback);

    // 定义时间同步订阅
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar0(nh, "/livox/lidar_192_168_147_231", 5);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar1(nh, "/livox/lidar_192_168_147_232", 5);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_lidar2(nh, "/livox/lidar_192_168_147_233", 5);

    // 使用 ApproximateTime 策略定义同步器
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(3), sub_lidar0, sub_lidar1, sub_lidar2);
    
    // 注册回调函数，当消息时间接近并准备同步时，该函数会被调用
    sync.registerCallback(boost::bind(&lidarcallback, _1, _2, _3));

    // 预分配内存
    lidar1PC->points.reserve(50000);
    lidar2PC->points.reserve(50000);
    lidar3PC->points.reserve(50000);
    lidarRawPC->points.reserve(150000);
    pc->points.reserve(150000);
    clusterIndices.reserve(20);

    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
};
